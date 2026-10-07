import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DDPM:
    def __init__(self, T=1000, schedule='linear', pred='eps', device='cuda'):
        assert pred in ('eps', 'x0', 'v')
        self.T = T
        self.pred = pred
        if schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, T)
        else: # cosine schedule
            s = 0.008
            steps = torch.arange(T + 1) / T
            f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            abar = f / f[0] # so that abar[0] = 1
            # abar_t = abar_{t-1} * a_t, a_t = abar_t / abar_{t-1}, beta = 1 - abar_t / abar_{t-1}
            betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999) 
        self.betas = betas.to(device)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)
        self.abar_prev = torch.cat([torch.tensor([1.0], device=device), self.abar[:-1]], dim=0)
        self.posterior_var = self.betas * (1 - self.abar_prev) / (1 - self.abar)
        # mean of q(x_{t-1} | x_t, x0) = coef1 * x0 + coef2 * xt
        self.posterior_coef1 = self.abar_prev.sqrt() * self.betas / (1 - self.abar)
        self.posterior_coef2 = self.alphas.sqrt() * (1 - self.abar_prev) / (1 - self.abar)

    @staticmethod
    def _extract(a, t, shape):
        """gather coefficients at t and reshape to (B,1,1,...) for broadcasting"""
        return a.gather(0, t).view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        abar_t = self._extract(self.abar, t, x0.shape)
        return x0 * abar_t.sqrt() + noise * (1 - abar_t).sqrt()

    def loss(self, model, x0, cond):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t, cond)
        if self.pred == 'eps':
            target = noise
        elif self.pred == 'x0':
            target = x0
        else:  # v = sqrt(abar) * eps - sqrt(1 - abar) * x0
            abar_t = self._extract(self.abar, t, x0.shape)
            target = abar_t.sqrt() * noise - (1 - abar_t).sqrt() * x0
        return F.mse_loss(pred, target)

    def _pred(self, model, x, t, cond, cfg_scale=1.0, null_label=10, abar_t=None):
        """convert the model output (eps / x0 / v) to (x0, eps), with x0 clipped to [-1, 1]
        for fractional timesteps t, the corresponding abar_t must be passed in directly"""
        if cond is None or cfg_scale == 1.0:
            out = model(x, t, cond)
        else:
            null_cond = torch.full_like(cond, null_label)
            out_c, out_u = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cond, null_cond])).chunk(2)
            out = out_u + cfg_scale * (out_c - out_u)

        if abar_t is None:
            abar_t = self._extract(self.abar, t, x.shape)
        if self.pred == 'eps':
            x0 = (x - (1 - abar_t).sqrt() * out) / abar_t.sqrt()
        elif self.pred == 'x0':
            x0 = out
        else:  # v
            x0 = abar_t.sqrt() * x - (1 - abar_t).sqrt() * out
        x0 = x0.clamp(-1, 1)
        eps = (x - abar_t.sqrt() * x0) / (1 - abar_t).sqrt()  # consistent with the clipped x0
        return x0, eps

    @torch.no_grad()
    def p_sample(self, model, xt, t, cond, cfg_scale=1.0):
        x0, _ = self._pred(model, xt, t, cond, cfg_scale)
        if t[0] == 0:  # at t=0, coef1=1 and coef2=0: return x0 directly to avoid float32 error
            return x0
        mean = self._extract(self.posterior_coef1, t, xt.shape) * x0 + self._extract(self.posterior_coef2, t, xt.shape) * xt
        var = self._extract(self.posterior_var, t, xt.shape)
        return mean + var.sqrt() * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, model, shape, cond, cfg_scale=1.0, device='cuda'):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond, cfg_scale)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, steps=50, eta=0.0, cfg_scale=1.0, device='cuda'):
        # trailing grid: split [0, T-1] into `steps` equal segments, call the network only at the high-noise end of each, and jump to abar=1 in the last one (no wasted call at t≈0)
        ts = torch.linspace(self.T - 1, 0, steps + 1, device=device)[:-1].long()
        x = torch.randn(shape, device=device)
        for i, t_cur in enumerate(ts):
            t = torch.full((shape[0],), int(t_cur), device=device, dtype=torch.long)
            abar_t = self._extract(self.abar, t, x.shape)
            abar_prev = self._extract(self.abar, torch.full_like(t, int(ts[i + 1])), shape) if i + 1 < steps else torch.ones_like(abar_t)
                
            x0_pred, eps = self._pred(model, x, t, cond, cfg_scale)
            sigma = eta * ((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)).sqrt()
            dir_xt = (1 - abar_prev - sigma ** 2).sqrt() * eps # direction pointing to x_t
            x = abar_prev.sqrt() * x0_pred + dir_xt + sigma * torch.randn_like(x)
        return x

    # ---------------- DPM-Solver++(2M) / UniPC ----------------
    # every step is a DDIM step; they differ only in which x0 is used for it; 1 network call per step
    # same time grid as DDIM, the last step goes to abar=1 (clean image)

    @staticmethod
    def _ddim_step(x, x0, abar_t, abar_s):
        """given x0, step deterministically from t to s (DDIM with eta=0)"""
        eps = (x - abar_t.sqrt() * x0) / (1 - abar_t).sqrt()
        return abar_s.sqrt() * x0 + (1 - abar_s).sqrt() * eps

    def _grid(self, steps, device):
        ts = torch.linspace(self.T - 1, 0, steps + 1)[:-1].long().tolist()  # trailing grid, same as ddim_sample
        abars = list(self.abar[ts]) + [torch.tensor(1.0, device=device)]
        lams = [0.5 * torch.log(a / (1 - a)) for a in abars[:-1]]  # lambda = log(alpha / sigma)
        return ts, abars, lams

    @torch.no_grad()
    def dpm_solver_sample(self, model, shape, cond, steps=20, cfg_scale=1.0, device='cuda'):
        """DPM-Solver++(2M): extrapolate x0 at the midpoint of this step using x0 from the previous step"""
        ts, abars, lams = self._grid(steps, device)
        x = torch.randn(shape, device=device)
        x0_prev = None
        for i, t in enumerate(ts):
            x0, _ = self._pred(model, x, torch.full((shape[0],), t, device=device, dtype=torch.long), cond, cfg_scale)
            if x0_prev is None or i == steps - 1:  # no history at the first step; the last step falls back to first order
                x0_eff = x0
            else:
                r = (lams[i] - lams[i - 1]) / (lams[i + 1] - lams[i])
                x0_eff = x0 + (x0 - x0_prev) / (2 * r)
            x = self._ddim_step(x, x0_eff, abars[i], abars[i + 1])
            x0_prev = x0
        return x

    @torch.no_grad()
    def unipc_sample(self, model, shape, cond, steps=20, cfg_scale=1.0, device='cuda'):
        """UniPC: the predictor (UniP) is the same as DPM-Solver++(2M);
        the corrector (UniC) redoes the step from x_t with the average of x0 at both ends, and both x0 are reused in the next step"""
        ts, abars, lams = self._grid(steps, device)
        pred = lambda x, t: self._pred(model, x, torch.full((shape[0],), t, device=device, dtype=torch.long), cond, cfg_scale)[0]
        x = torch.randn(shape, device=device)
        x0_cur, x0_prev = pred(x, ts[0]), None
        for i in range(steps):
            # predictor (UniP)
            if x0_prev is None or i == steps - 1:
                x0_eff = x0_cur
            else:
                r = (lams[i] - lams[i - 1]) / (lams[i + 1] - lams[i])
                x0_eff = x0_cur + (x0_cur - x0_prev) / (2 * r)
            x_pred = self._ddim_step(x, x0_eff, abars[i], abars[i + 1])
            if i == steps - 1:  # last step: nothing to reuse afterwards, so no correction
                return x_pred

            x0_new = pred(x_pred, ts[i + 1])  # the only network call of this step, also reused by the next step
            # corrector (UniC): average x0 at both ends and redo this step from x_t
            x = self._ddim_step(x, 0.5 * (x0_cur + x0_new), abars[i], abars[i + 1])
            x0_prev, x0_cur = x0_cur, x0_new

    def _lam_to_t(self, lam):
        """lambda -> fractional timestep, by linear interpolation on the discrete lambda table (lambda decreases monotonically with t)"""
        lam_all = 0.5 * torch.log(self.abar / (1 - self.abar))
        j = (lam_all > lam).sum().clamp(1, self.T - 1)
        w = (lam_all[j - 1] - lam) / (lam_all[j - 1] - lam_all[j])
        return (j - 1 + w).item()

    @torch.no_grad()
    def dpm_solver2_sample(self, model, shape, cond, steps=20, cfg_scale=1.0, device='cuda'):
        """DPM-Solver-2 (eps form, single-step second order): each step first probes to the lambda midpoint and then takes the full step with the midpoint eps
        2 network calls per step (1 for the last step), 2 * steps - 1 in total"""
        ts, abars, lams = self._grid(steps, device)
        # DDIM step in eps form: compute x0 from eps, then re-noise to s with the same eps
        eps_step = lambda x, eps, a_t, a_s: a_s.sqrt() * (x - (1 - a_t).sqrt() * eps) / a_t.sqrt() + (1 - a_s).sqrt() * eps
        pred = lambda x, t, a: self._pred(model, x, torch.full((shape[0],), t, device=device), cond, cfg_scale, abar_t=a)
        x = torch.randn(shape, device=device)
        for i, t in enumerate(ts):
            x0, eps_t = pred(x, t, abars[i])
            if i == steps - 1:  # the last step goes to the clean image where the lambda midpoint does not exist, so use first order
                return x0
            lam_m = 0.5 * (lams[i] + lams[i + 1])
            abar_m = torch.sigmoid(2 * lam_m)
            x_m = eps_step(x, eps_t, abars[i], abar_m)                     # step 1: probe to the midpoint
            _, eps_m = pred(x_m, self._lam_to_t(lam_m), abar_m)            # step 2: noise estimate at the midpoint
            x = eps_step(x, eps_m, abars[i], abars[i + 1])                 # step 3: full step from x_t with eps_m
