import torch
import torch.nn.functional as F


class TDM:
    """TDM (Luo et al. 2025) distillation with the RectifiedFlow convention: x_t = (1 - t) * x0 + t * x1, networks output the velocity v.
    The student G is a K-step Euler sampler on t_k = 1 - k/K. Each step's output x_{t_{k+1}} is matched to the teacher only at
    tau in [t_{k+1}, t_k], so the whole trajectory (not just the final sample) is distilled. Data-free, no GAN."""

    def __init__(self, num_steps=1, t_scale=1000.0, cfg_scale=3.0, null_label=10, t_min=0.02, t_max=0.98, huber_c=0.00054 * 784 ** 0.5):
        self.num_steps = num_steps
        self.ts = torch.linspace(1.0, 0.0, num_steps + 1)
        self.t_scale = t_scale
        self.cfg_scale, self.null_label = cfg_scale, null_label
        self.t_min, self.t_max = t_min, t_max
        self.huber_c = huber_c

    def _x0(self, model, x, t, cond, cfg_scale=1.0):
        tt = t * self.t_scale
        if cfg_scale == 1.0:
            v = model(x, tt, cond)
        else:
            v_c, v_u = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([cond, torch.full_like(cond, self.null_label)])).chunk(2)
            v = v_u + cfg_scale * (v_c - v_u)
        return x - t.view(-1, 1, 1, 1) * v

    def sample_step(self, G, z, cond, k=None, u=None, noise=None):
        """Pick a random step k of the student trajectory, return (x0, x_tau, tau, k).
        Only step k is backpropagated through; the rollout to x_{t_k} runs without grad."""
        B = z.shape[0]
        ts = self.ts.to(z.device)
        k = torch.randint(0, self.num_steps, (B,), device=z.device) if k is None else k
        with torch.no_grad():
            xs, x = [z], z
            for i in range(self.num_steps - 1):
                x = x + (ts[i + 1] - ts[i]) * G(x, ts[i].expand(B) * self.t_scale, cond)
                xs.append(x)
            x_t = torch.stack(xs)[k, torch.arange(B)]
        t, t_next = ts[k], ts[k + 1]
        tb, tnb = t.view(-1, 1, 1, 1), t_next.view(-1, 1, 1, 1)

        v = G(x_t, t * self.t_scale, cond)
        x0 = x_t - tb * v
        x_next = x_t + (tnb - tb) * v

        # forward-noise x_{t_next} to tau ~ U[t_next, t] with the RF transition kernel
        lo, hi = t_next.clamp(min=self.t_min), t.clamp(max=self.t_max)
        tau = lo + (torch.rand(B, device=z.device) if u is None else u) * (hi - lo)
        noise = torch.randn_like(z) if noise is None else noise
        a = ((1 - tau) / (1 - t_next)).view(-1, 1, 1, 1)
        std = (tau.view(-1, 1, 1, 1) ** 2 - (a * tnb) ** 2).clamp(min=0).sqrt()
        x_tau = a * x_next.detach() + std * noise
        return x0, x_tau, tau, k

    def loss_dm(self, x0, x_tau, tau, teacher, fake, cond):
        """Per-sample pseudo-Huber on x0 - sg(x0 + x0_real - x0_fake); its gradient ≈ d / ||d|| is normalized per sample."""
        with torch.no_grad():
            x0_real = self._x0(teacher, x_tau, tau, cond, self.cfg_scale)
            x0_fake = self._x0(fake, x_tau, tau, cond)
            target = x0 + x0_real - x0_fake
        sq = ((x0 - target) ** 2).sum(dim=(1, 2, 3))
        return torch.sqrt(sq + self.huber_c ** 2) - self.huber_c

    def loss_fake(self, fake, x0, x_tau, tau, cond):
        """Fake score: flow matching on the student's trajectory samples, regressing the student's x0."""
        x0, x_tau = x0.detach(), x_tau.detach()
        return F.mse_loss(fake(x_tau, tau * self.t_scale, cond), (x_tau - x0) / tau.view(-1, 1, 1, 1))

    @torch.no_grad()
    def denoise_err(self, model, x0, x_tau, tau, cond):
        return F.mse_loss(self._x0(model, x_tau, tau, cond), x0)

    @torch.no_grad()
    def sample(self, G, shape, cond, device='cuda'):
        x = torch.randn(shape, device=device)
        ts = self.ts.to(device)
        for i in range(self.num_steps):
            x = x + (ts[i + 1] - ts[i]) * G(x, ts[i].expand(shape[0]) * self.t_scale, cond)
        return x
