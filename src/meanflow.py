import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


class MeanFlow:
    """Same convention as RectifiedFlow: z_t = (1 - t) * x0 + t * x1, t=0 is data and t=1 is noise.
    The model predicts the average velocity u(z_t, r, t) over [r, t], conditioned on (t, t - r); at r = t, u is the instantaneous velocity v.
    MeanFlow identity: u = v - (t - r) * du/dt, du/dt = v * ∂u/∂z + ∂u/∂t (total derivative, r fixed)"""

    def __init__(self, t_scale=1.0, p_mean=-0.4, p_std=1.0, ratio_r_neq_t=0.25, p=1.0, c=1e-3, cfg_scale=1.0, null_label=10):
        # as in the official implementation, t and h enter the time embedding in [0,1] without the x1000 scaling used by RF:
        # the target contains du/dt, and with x1000 the high-frequency sinusoidal components amplify ∂u/∂h ~1000x, which dominates the target and makes training diverge
        self.t_scale = t_scale
        # training-time CFG (ω): guidance is distilled into the model, so sampling stays 1 step without CFG; requires label dropout in the training script
        self.cfg_scale, self.null_label = cfg_scale, null_label
        self.p_mean, self.p_std = p_mean, p_std  # logit-normal sampling parameters of (t, r)
        self.ratio_r_neq_t = ratio_r_neq_t  # fraction of samples with r != t; the rest have r = t and reduce to flow matching
        self.p, self.c = p, c  # adaptive loss weight: w = 1 / (||Δ||² + c)^p

    def u(self, model, z, r, t, cond):
        return model(z, t * self.t_scale, cond, h=(t - r) * self.t_scale)

    def sample_tr(self, B, device):
        s = torch.sigmoid(torch.randn(B, 2, device=device) * self.p_std + self.p_mean)
        t, r = s.max(dim=1).values, s.min(dim=1).values
        r = torch.where(torch.rand(B, device=device) < self.ratio_r_neq_t, r, t)
        return t, r

    def loss(self, model, x0, cond):
        t, r = self.sample_tr(x0.shape[0], x0.device)
        x1 = torch.randn_like(x0)
        tb = t.view(-1, 1, 1, 1)
        z = (1 - tb) * x0 + tb * x1
        v = x1 - x0  # conditional velocity, same as RF
        if self.cfg_scale != 1.0:
            # guided velocity ṽ = ω v + (1 - ω) u(z, t, t | ∅): the unconditional instantaneous velocity is replaced by the model's own prediction at r=t
            # samples dropped to the null label learn the unconditional velocity without guidance
            uncond = (cond == self.null_label).view(-1, 1, 1, 1)
            with torch.no_grad():
                u_unc = self.u(model, z, t, t, torch.full_like(cond, self.null_label))
            v = torch.where(uncond, v, self.cfg_scale * v + (1 - self.cfg_scale) * u_unc)

        # du/dt via JVP with tangents (dz, dr, dt) = (v, 0, 1) (v is ṽ with CFG); t_scale is applied inside self.u, so the derivative is w.r.t. the raw t
        # the target is stop-grad anyway, so the JVP runs under no_grad without building a graph (otherwise the tangent activations are also stored, ~5x memory)
        # the default SDPA kernels do not support forward-mode AD, so the JVP uses the MATH backend
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            _, dudt = torch.func.jvp(
                lambda z, r, t: self.u(model, z, r, t, cond),
                (z, r, t),
                (v, torch.zeros_like(r), torch.ones_like(t)),
            )
        u_tgt = v - (t - r).view(-1, 1, 1, 1) * dudt
        u = self.u(model, z, r, t, cond)  # regular forward (default SDPA); gradients only flow through here

        err = ((u - u_tgt) ** 2).flatten(1).sum(1)
        w = 1.0 / (err.detach() + self.c) ** self.p
        mse = err.detach().mean() / x0[0].numel()  # for logging only: the loss is ~constant after adaptive weighting and shows no trend
        return (w * err).mean(), mse

    @torch.no_grad()
    def sample(self, model, shape, cond, ts=(1.0, 0.0), device='cuda'):
        """ts=(1.0, 0.0) is one-step generation; multistep follows ts with z_r = z_t - (t - r) * u(z_t, r, t), deterministic, no re-noising"""
        z = torch.randn(shape, device=device)
        for t, r in zip(ts[:-1], ts[1:]):
            tt = torch.full((shape[0],), float(t), device=device)
            rr = torch.full((shape[0],), float(r), device=device)
            z = z - (t - r) * self.u(model, z, rr, tt, cond)
        return z
