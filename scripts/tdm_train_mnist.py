import copy
import os
import sys

import torch
from torchvision.utils import save_image

from dit import DiT
from tdm import TDM

device = "cuda"
num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2  # student steps K
epochs = 20
iters_per_epoch = 70000 // 256  # data-free, epoch length matched to MNIST
batch_size = 256
lr_fake = 2e-5
lr_G = 4e-6
num_classes = 10
cfg_scale = 3.0
dfake_gen_update_ratio = 5
max_grad_norm = 10.0
teacher_ckpt = "results/train_mnist_rf/model.pt"
out_dir = f"results/train_mnist_tdm_{num_steps}step"
os.makedirs(out_dir, exist_ok=True)

teacher = DiT(num_classes=num_classes).to(device)
teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
teacher.eval().requires_grad_(False)
G = copy.deepcopy(teacher).train().requires_grad_(True)
fake = copy.deepcopy(teacher).train().requires_grad_(True)

tdm = TDM(num_steps=num_steps, cfg_scale=cfg_scale, null_label=num_classes)
opt_G = torch.optim.AdamW(G.parameters(), lr=lr_G)
opt_fake = torch.optim.AdamW(fake.parameters(), lr=lr_fake)

g = torch.Generator(device=device).manual_seed(0)
probe_z = torch.randn(batch_size, 1, 28, 28, device=device, generator=g)
probe_y = torch.arange(batch_size, device=device) % num_classes
probe_k = torch.arange(batch_size, device=device) % num_steps
probe_u = torch.rand(batch_size, device=device, generator=g)
probe_noise = torch.randn(batch_size, 1, 28, 28, device=device, generator=g)


@torch.no_grad()
def probe(model):
    x0, x_tau, tau, _ = tdm.sample_step(G, probe_z, probe_y, probe_k, probe_u, probe_noise)
    return tdm.denoise_err(model, x0, x_tau, tau, probe_y).item()


it = 0
for epoch in range(epochs):
    for step in range(iters_per_epoch):
        z = torch.randn(batch_size, 1, 28, 28, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

        if it % dfake_gen_update_ratio == 0:
            x0, x_tau, tau, k = tdm.sample_step(G, z, y)
            loss_dm = tdm.loss_dm(x0, x_tau, tau, teacher, fake, y)
            loss_dm_k = [loss_dm[k == i].mean().item() for i in range(num_steps)]
            opt_G.zero_grad()
            loss_dm.mean().backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_grad_norm)
            opt_G.step()
            err_fake_after_G = probe(fake)

        with torch.no_grad():
            x0, x_tau, tau, _ = tdm.sample_step(G, z, y)
        loss_fake = tdm.loss_fake(fake, x0, x_tau, tau, y)
        opt_fake.zero_grad()
        loss_fake.backward()
        torch.nn.utils.clip_grad_norm_(fake.parameters(), max_grad_norm)
        opt_fake.step()

        if it % dfake_gen_update_ratio == dfake_gen_update_ratio - 1 and step // dfake_gen_update_ratio % 20 == 0:
            err_fake, err_teacher = probe(fake), probe(teacher)
            print(f"epoch {epoch} step {step} loss_dm {' '.join(f'{l:.4f}' for l in loss_dm_k)} loss_fake {loss_fake.item():.4f} "
                  f"lag {err_fake_after_G / err_fake:.3f} fake/teacher {err_fake / err_teacher:.3f}")
        it += 1

    if epoch % 5 == 0 or epoch == epochs - 1:
        G.eval()
        y = torch.arange(num_classes, device=device).repeat_interleave(10)
        samples = tdm.sample(G, (len(y), 1, 28, 28), cond=y, device=device)
        save_image(samples, f"{out_dir}/epoch_{epoch:03d}_{num_steps}step.png", nrow=10, normalize=True, value_range=(-1, 1))
        torch.save(G.state_dict(), f"{out_dir}/model.pt")
        G.train()
