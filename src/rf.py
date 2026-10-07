import torch
import torch.nn.functional as F
from tqdm import tqdm


class RectifiedFlow:
    """x_t = (1 - t) * x0 + t * x1, t=0 is data and t=1 is noise; the model predicts the velocity v = x1 - x0"""

    def __init__(self, t_sampling="logit_normal", shift=1.0, t_scale=1000.0):
        self.t_sampling = t_sampling
        self.shift = shift  # resolution-dependent timestep shift (SD3)
        self.t_scale = t_scale  # scale t ∈ [0,1] to [0,1000] before feeding the model, to match DiT's sinusoidal time embedding

    def _shift_t(self, t):
        return self.shift * t / (1 + (self.shift - 1) * t)
    
    def sample_t(self, B, device):
        if self.t_sampling == "uniform":
            t = torch.rand(B, device=device)
        else:  # logit-normal: sample more intermediate timesteps, where the loss is hardest
            t = torch.sigmoid(torch.randn(B, device=device))
        if self.shift != 1.0:  # shift towards higher noise (higher resolutions need more noise to destroy information)
            # endpoints t=0->0, t=1->1 are kept and the map is monotonic, it only redistributes density in between; longer sequences need a larger shift
            # t = self.shift * t / (1 + (self.shift - 1) * t)
            t = self._shift_t(t)
        return t

    def loss(self, model, x0, cond):
        t = self.sample_t(x0.shape[0], x0.device)
        x1 = torch.randn_like(x0)
        tb = t.view(-1, *([1] * (x0.ndim - 1)))
        xt = (1 - tb) * x0 + tb * x1
        v_pred = model(xt, t * self.t_scale, cond)
        return F.mse_loss(v_pred, x1 - x0)  # v = dx_t/dt = x1 - x0

    def _v(self, model, x, t, cond, cfg_scale=1.0, null_label=10):
        tb = torch.full((x.shape[0],), float(t) * self.t_scale, device=x.device)
        if cond is None or cfg_scale == 1.0:
            return model(x, tb, cond)
        null_cond = torch.full_like(cond, null_label)
        v_c, v_u = model(torch.cat([x, x]), torch.cat([tb, tb]), torch.cat([cond, null_cond])).chunk(2)
        return v_u + cfg_scale * (v_c - v_u)

    @torch.no_grad()
    def sample(self, model, shape, cond, steps=50, cfg_scale=1.0, device='cuda'):
        x = torch.randn(shape, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1)
        
        if self.shift != 1.0:
            ts = self._shift_t(ts)
        
        for i in tqdm(range(steps)):
            t, t_next = ts[i], ts[i + 1]
            v = self._v(model, x, t, cond, cfg_scale)
            x = x + (t_next - t) * v  # dt is negative: move along -v towards the data
        return x

    @torch.no_grad()
    def sample_heun(self, model, shape, cond, steps=25, cfg_scale=1.0, device='cuda'):
        x = torch.randn(shape, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1)
        
        if self.shift != 1.0:
            ts = self._shift_t(ts)

        for i in range(steps):
            t, t_next = ts[i], ts[i + 1]
            dt = t_next - t
            v1 = self._v(model, x, t, cond, cfg_scale)
            if t_next == 0:  # Euler for the last step
                x = x + dt * v1
                break
            x_pred = x + dt * v1
            v2 = self._v(model, x_pred, t_next, cond, cfg_scale)
            x = x + dt * 0.5 * (v1 + v2)
        return x
