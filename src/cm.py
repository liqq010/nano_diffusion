import math
import torch


class ConsistencyModel:
    """Same convention as RectifiedFlow: x_t = (1 - t) * x0 + t * x1, t=0 is data and t=1 is noise.
    The model still outputs a velocity v; the consistency function f(x_t, t) = x_t - t * v predicts x0, and f = x at t=0, so the boundary condition holds by construction."""

    def __init__(self, t_scale=1000.0, s0=10, s1=1280, total_iters=None):
        self.t_scale = t_scale  # same as RF: t is scaled to [0, 1000] before being fed to the model
        self.s0, self.s1 = s0, s1  # curriculum of the number of discretization steps N: doubled from s0 up to s1 (iCT)
        self.total_iters = total_iters

    def num_steps(self, it):
        """iCT exponential curriculum: N doubles every K' iterations; large Δt early (stable), small Δt late (low bias)"""
        n_stages = int(math.log2(self.s1 / self.s0)) + 1
        stage_len = max(self.total_iters // n_stages, 1)
        return min(self.s0 * 2 ** (it // stage_len), self.s1)

    # the model predicts the average velocity from x_t to the endpoint
    # f is the predicted endpoint of the ODE
    def f(self, model, x, t, cond):
        """t: (B,) ∈ [0,1]"""
        v = model(x, t * self.t_scale, cond)
        return x - t.view(-1, *([1] * (x.ndim - 1))) * v

    def loss_ct(self, model, x0, cond, it):
        B = x0.shape[0]
        N = self.num_steps(it)
        # iCT lognormal: ln σ ~ N(-1.1, 2²) with σ = t / (1 - t), i.e. t = sigmoid(ln σ); then snapped to the grid t_i = i/N, t' = t_{i-1}
        # much wider than RF's logit-normal (0, 1): enough samples at high noise, so training does not collapse when N switches
        t = torch.sigmoid(torch.randn(B, device=x0.device) * 2.0 - 1.1)
        i = torch.ceil(t * N).clamp(1, N)
        t, tp = i / N, (i - 1) / N
        x1 = torch.randn_like(x0)
        tb, tpb = t.view(-1, 1, 1, 1), tp.view(-1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * x1
        xtp = (1 - tpb) * x0 + tpb * x1  # same (x0, x1) pair: equivalent to one Euler step with the conditional velocity x1 - x0

        pred = self.f(model, xt, t, cond)
        with torch.no_grad():  # teacher = stopgrad(current model); iCT found an EMA teacher to be harmful
            target = self.f(model, xtp, tp, cond)  # at tp=0 the target is the true x0

        # weight 1/Δt = N compensates for the difference shrinking with Δt
        return (N * self._pseudo_huber(pred, target)).mean(), N

    def loss_cd(self, model, teacher, x0, cond, N=20, cfg_scale=1.0):
        """Key difference from CT: x_{t'} is obtained by one Heun step of the RF teacher from x_t (instead of interpolating the same (x0, x1) pair).
        With cfg_scale > 1 the teacher uses the guided velocity, so guidance is distilled into the student (no CFG needed at sampling)."""
        B = x0.shape[0]
        # original CD: fixed N, index i sampled uniformly from {1..N}
        i = torch.randint(1, N + 1, (B,), device=x0.device).float()
        t, tp = i / N, (i - 1) / N
        x1 = torch.randn_like(x0)
        tb = t.view(-1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * x1

        with torch.no_grad():
            dt = (tp - t).view(-1, 1, 1, 1)
            v1 = self._teacher_v(teacher, xt, t, cond, cfg_scale)
            x_euler = xt + dt * v1
            v2 = self._teacher_v(teacher, x_euler, tp, cond, cfg_scale)
            xtp = xt + dt * 0.5 * (v1 + v2)
            target = self.f(model, xtp, tp, cond)  # stopgrad(current model), same as CT

        pred = self.f(model, xt, t, cond)
        return self._pseudo_huber(pred, target).mean()

    def _teacher_v(self, teacher, x, t, cond, cfg_scale, null_label=10):
        if cfg_scale == 1.0:
            return teacher(x, t * self.t_scale, cond)
        null_cond = torch.full_like(cond, null_label)
        v_c, v_u = teacher(torch.cat([x, x]), torch.cat([t, t]) * self.t_scale, torch.cat([cond, null_cond])).chunk(2)
        return v_u + cfg_scale * (v_c - v_u)

    @staticmethod
    def _pseudo_huber(pred, target):
        # c = 0.00054 * sqrt(d) (iCT)
        c = 0.00054 * math.sqrt(pred[0].numel())
        return torch.sqrt(((pred - target) ** 2).flatten(1).sum(1) + c ** 2) - c

    @torch.no_grad()
    def sample(self, model, shape, cond, ts=(1.0,), device='cuda'):
        """ts=(1.0,) is one-step generation; multistep: predict x0 -> re-noise to the next t -> predict again.
        No CFG: CT has no teacher so guidance cannot be done correctly; CD already has guidance distilled in during training."""
        x = torch.randn(shape, device=device)
        for k, t in enumerate(ts):
            x0 = self.f(model, x, torch.full((shape[0],), float(t), device=device), cond)
            if k + 1 < len(ts):
                x = (1 - ts[k + 1]) * x0 + ts[k + 1] * torch.randn_like(x0)
        return x0
