import copy
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from mnist_dataset import MNISTDataset
from dit import DiT
from dmd import DMD2, DiscHead

device = "cuda"
epochs = 50
batch_size = 256
lr = 2e-5  # same for G and the fake score incl. the discriminator head (2e-6 in the official ImageNet config)
num_classes = 10
cfg_scale = 3.0  # teacher guidance, distilled into G
dfake_gen_update_ratio = 5  # TTUR: the fake score is updated every step, G every 5 steps
gen_gan_weight = 3e-3  # GAN loss weight for G (official gen_cls_loss_weight)
fake_gan_weight = 1e-2  # discriminator loss weight (official cls_loss_weight)
max_grad_norm = 10.0
teacher_ckpt = "results/train_mnist_rf/model.pt"
out_dir = "results/train_mnist_dmd2"
os.makedirs(out_dir, exist_ok=True)

dataset = MNISTDataset(root="data/mnist/mnist")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

teacher = DiT(num_classes=num_classes).to(device)
teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
teacher.eval().requires_grad_(False)
G = copy.deepcopy(teacher).train().requires_grad_(True)  # generator and fake score are both initialized from the teacher
fake = copy.deepcopy(teacher).train().requires_grad_(True)
head = DiscHead().to(device)

dmd = DMD2(fake=fake, cfg_scale=cfg_scale, null_label=num_classes)
opt_G = torch.optim.AdamW(G.parameters(), lr=lr)
opt_fake = torch.optim.AdamW(list(fake.parameters()) + list(head.parameters()), lr=lr)

# fixed probe batch for monitoring: fixed z / labels / t / noise so metrics are comparable across steps
g = torch.Generator(device=device).manual_seed(0)
probe_z = torch.randn(batch_size, 1, 28, 28, device=device, generator=g)
probe_y = torch.arange(batch_size, device=device) % num_classes
probe_t = torch.empty(batch_size, device=device).uniform_(0.02, 0.98, generator=g)
probe_noise = torch.randn(batch_size, 1, 28, 28, device=device, generator=g)


@torch.no_grad()
def probe(model):
    """denoising error of model on the current G samples (MSE in x0 space)"""
    return dmd.denoise_err(model, dmd.generate(G, probe_z, probe_y), probe_t, probe_noise, probe_y).item()


it = 0
for epoch in range(epochs):
    for step, (x_real, y) in enumerate(loader):
        x_real, y = x_real.to(device), y.to(device)
        z = torch.randn_like(x_real)

        if it % dfake_gen_update_ratio == 0:  # generator: distribution matching + GAN
            x = dmd.generate(G, z, y)
            loss_dm, grad_norm = dmd.loss_dm(x, teacher, fake, y)
            loss_g_gan = dmd.loss_gan_g(fake, head, x, y)
            opt_G.zero_grad()
            (loss_dm + gen_gan_weight * loss_g_gan).backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_grad_norm)
            opt_G.step()
            err_fake_after_G = probe(fake)  # G just updated and fake not yet caught up: error at the largest lag

        # fake score + discriminator: fit G's current output distribution and tell real data from G outputs
        with torch.no_grad():
            x = dmd.generate(G, z, y)
        loss_fake = dmd.loss_fake(fake, x, y)
        loss_d, p_real, p_fake = dmd.loss_gan_d(fake, head, x_real, x, y, y)
        opt_fake.zero_grad()  # also clears the gradients that G's GAN loss above left on fake/head
        (loss_fake + fake_gan_weight * loss_d).backward()
        torch.nn.utils.clip_grad_norm_(list(fake.parameters()) + list(head.parameters()), max_grad_norm)
        opt_fake.step()

        # monitoring: at the end of each TTUR cycle (before the next G update), record the error after fake has caught up
        # lag = before / after catching up, ≈1 means fake keeps up with G; fake/teacher < 1 means fake has learned G's distribution (otherwise the DM gradient is unreliable)
        if it % dfake_gen_update_ratio == dfake_gen_update_ratio - 1 and step // dfake_gen_update_ratio % 20 == 0:
            err_fake = probe(fake)
            err_teacher = probe(teacher)
            print(f"epoch {epoch} step {step} loss_dm {loss_dm.item():.4f} grad_norm {grad_norm.item():.4f} "
                  f"loss_g_gan {loss_g_gan.item():.4f} loss_fake {loss_fake.item():.4f} loss_d {loss_d.item():.4f} "
                  f"p_real {p_real.item():.3f} p_fake {p_fake.item():.3f} "
                  f"lag {err_fake_after_G / err_fake:.3f} err_fake {err_fake:.4f} err_teacher {err_teacher:.4f} fake/teacher {err_fake / err_teacher:.3f}")
        it += 1

    # G is one-step, no CFG
    if epoch % 5 == 0 or epoch == epochs - 1:
        G.eval()
        y = torch.arange(num_classes, device=device).repeat_interleave(10)
        samples = dmd.sample(G, (len(y), 1, 28, 28), cond=y, device=device)
        save_image(samples, f"{out_dir}/epoch_{epoch:03d}_1step.png", nrow=10, normalize=True, value_range=(-1, 1))
        torch.save(G.state_dict(), f"{out_dir}/model.pt")
        G.train()
