import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscHead(nn.Module):
    """GAN discriminator head (after the official cls_pred_branch): applied to the token features of the last DiT block of the fake score network;
    (B, T, D) is reshaped to a (B, D, h, w) feature map and convolved down to 1x1, giving one logit"""

    def __init__(self, hidden_size=256, grid=14):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.GroupNorm(32, hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=grid // 2),  # 7x7 -> 1x1
            nn.GroupNorm(32, hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, 1, kernel_size=1),
        )

    def forward(self, tokens):
        B, T, D = tokens.shape
        return self.net(tokens.transpose(1, 2).reshape(B, D, self.grid, self.grid)).view(B)


class DMD2:
    """DMD2 (Yin et al. 2024) distillation with the RectifiedFlow convention: x_t = (1 - t) * x0 + t * x1, t=0 is data and t=1 is noise, networks output the velocity v.
    teacher (frozen RF, real score), generator G (one step x = z - v_G(z, t=1)), fake score (fits G's output distribution online, with the GAN discriminator head).
    G loss = distribution matching (difference of the teacher and fake-score x0 predictions on noised samples, i.e. the gradient of KL(p_fake || p_real)) + GAN;
    fake score loss = flow matching on G's outputs + discriminator classification (real data vs G outputs)."""

    def __init__(self, fake=None, t_scale=1000.0, cfg_scale=3.0, null_label=10, t_min=0.02, t_max=0.98):
        self.t_scale = t_scale  # same as RF
        self.cfg_scale, self.null_label = cfg_scale, null_label  # teacher guidance, distilled into G
        self.t_min, self.t_max = t_min, t_max  # t range of the DM loss (2%-98% of the timesteps in the official code)
        if fake is not None:  # forward hook grabs the last-block features of the fake network for the discriminator, without modifying DiT
            fake.blocks[-1].register_forward_hook(lambda m, i, o: setattr(self, "_feat", o))

    def _x0(self, model, x, t, cond, cfg_scale=1.0):
        """x0 = x_t - t * v; t: (B,)"""
        tt = t * self.t_scale
        if cfg_scale == 1.0:
            v = model(x, tt, cond)
        else:
            v_c, v_u = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([cond, torch.full_like(cond, self.null_label)])).chunk(2)
            v = v_u + cfg_scale * (v_c - v_u)
        return x - t.view(-1, 1, 1, 1) * v

    def _noise(self, x, t):
        return (1 - t.view(-1, 1, 1, 1)) * x + t.view(-1, 1, 1, 1) * torch.randn_like(x)

    def generate(self, G, z, cond):
        """one-step generation: predict x0 directly from pure noise at t=1"""
        return self._x0(G, z, torch.ones(z.shape[0], device=z.device), cond)

    def loss_dm(self, x, teacher, fake, cond):
        """distribution matching loss, x is G's output (with grad). Its gradient w.r.t. x is proportional to grad = (x0_fake - x0_real) / w, and -grad pushes x towards the teacher distribution"""
        with torch.no_grad():
            t = torch.empty(x.shape[0], device=x.device).uniform_(self.t_min, self.t_max)
            xt = self._noise(x, t)
            x0_real = self._x0(teacher, xt, t, cond, self.cfg_scale)
            x0_fake = self._x0(fake, xt, t, cond)
            w = (x - x0_real).abs().mean(dim=(1, 2, 3), keepdim=True)  # per-sample normalization, so gradients have a similar magnitude across t
            grad = torch.nan_to_num((x0_fake - x0_real) / w)
        return 0.5 * F.mse_loss(x, (x - grad).detach()), grad.norm() / x.shape[0] ** 0.5

    def loss_fake(self, fake, x, cond):
        """fake score: flow matching on G's outputs (detached), t ~ U[0, 1] covers all noise levels"""
        x = x.detach()
        t = torch.rand(x.shape[0], device=x.device)
        x1 = torch.randn_like(x)
        xt = (1 - t.view(-1, 1, 1, 1)) * x + t.view(-1, 1, 1, 1) * x1
        return F.mse_loss(fake(xt, t * self.t_scale, cond), x1 - x)

    def _logits(self, fake, head, x, cond):
        """diffusion GAN: noise with a random t ~ U[0, 1], then pass the fake network features through the discriminator head"""
        t = torch.rand(x.shape[0], device=x.device)
        fake(self._noise(x, t), t * self.t_scale, cond)
        return head(self._feat)

    def loss_gan_d(self, fake, head, x_real, x_fake, cond_real, cond_fake):
        """discriminator (fake network + head): real data as real, G outputs as fake; also returns the mean probabilities D(real), D(fake) (both ~0.5 at equilibrium)"""
        l_real = self._logits(fake, head, x_real, cond_real)
        l_fake = self._logits(fake, head, x_fake.detach(), cond_fake)
        loss = (F.softplus(-l_real) + F.softplus(l_fake)).mean()
        return loss, torch.sigmoid(l_real).mean().detach(), torch.sigmoid(l_fake).mean().detach()

    def loss_gan_g(self, fake, head, x, cond):
        """generator: non-saturating GAN loss, gradients flow back to x through the fake network and head (their gradients are zeroed before their own update)"""
        return F.softplus(-self._logits(fake, head, x, cond)).mean()

    @torch.no_grad()
    def denoise_err(self, model, x, t, noise, cond):
        """for monitoring: noise x with a fixed (t, noise), predict x0 with model, and return the MSE to x"""
        xt = (1 - t.view(-1, 1, 1, 1)) * x + t.view(-1, 1, 1, 1) * noise
        return F.mse_loss(self._x0(model, xt, t, cond), x)

    @torch.no_grad()
    def sample(self, G, shape, cond, device='cuda'):
        return self.generate(G, torch.randn(shape, device=device), cond)
