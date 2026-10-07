# nano_diffusion
From 1000-step DDPM to 1-step generators: clean, minimal PyTorch implementations of diffusion and flow models  

### 1. how to run: 
```bash
# get mnist dataset
git clone https://huggingface.co/datasets/ylecun/mnist data/mnist   
# create synthetic outlined mnist for LoRA
python scripts/create_mnist_outlined.py  
# train / generate 
PYTHONPATH=src python scripts/rf_train_mnist.py  
PYTHONPATH=src python scripts/rf_test_mnist.py 
```
### 2. covered
- [x] DDPM eps/x0/v-prediction, CFG, linear/cosine schedule, trailing, zero terminal SNR, min-SNR-𝛄
- [x] DDIM / DPM-Solver-2 / DPM++ / UniPC
- [x] Rectified Flow, logit-normal, timestep shift  
- [x] Consistency Models (CT / CD)
- [x] MeanFlow
- [x] DMD2  
- [x] LoRA finetune  

|  |  |
|---|---|
| `src/dit.py` | DiT backbone |
| `src/ddpm.py` | DDPM (eps / x0 / v), DDPM / DDIM / DPM-Solver-2 / DPM++(2M) / UniPC samplers |
| `src/rf.py` | Rectified Flow, Euler / Heun samplers |
| `src/cm.py` | Consistency Models, CT (iCT) / CD |
| `src/meanflow.py` | MeanFlow |
| `src/dmd.py` | DMD2, distribution matching + GAN losses |
| `src/lora.py` | LoRA layers |


### 3. ablation studies  
experiment settings:  
DiT (28x28 input, patch 2, hidden 256, depth 8, 16 heads, 9.79M params)  

#### 3.1 linear vs cosine noise schedule
![noise_schedule_cmp](figs/ddpm_linear_cosine_noise_schedule.png)  
#### 3.2 DDPM eps / x0 / v-prediction  
training loss curve  
![loss_curve](figs/ddpm_loss_curve.png)  
samples generated during training  
![train_progress](figs/ddpm_train_progress.png)  
comparison of different solvers:  DDIM / DPM++ / UniPC  
![solver_cmp](figs/ddpm_eps_solvers_cmp.png)    
#### 3.3 Rectified Flow  
comparison between Euler and Heun   
![rf_solver_cmp](figs/rf_solvers_cmp.png)
#### 3.4 Rectified Flow w/o CFG  
comparison between samples generated w/o classifier-free guidance   
![rf_cfg_cmp](figs/rf_cfg_cmp.png)
#### 3.5 Consistency Models, CT(iCT) vs CD  
comparison between consistency training (CT) and consistency distillation (CD)  
![ct_cd_cmp](figs/cm_ct_cd_cmp.png)
#### 3.6 MeanFlow
samples generated during training  
![train_progress](figs/meanflow_cfg3_train_progress.png)
#### 3.7 DMD2 distillation
one-step generator distilled from the RF teacher (cfg=3) with distribution matching + GAN loss, TTUR 5:1, trained for 20 epochs  
training metrics:  
- grad_norm: norm of the distribution matching gradient (x0_fake - x0_real) / w  
- loss_fake: flow matching loss of the fake score on generator samples  
- lag: denoise error of the fake score right after a generator update / after 5 fake updates (≈1 means the fake score keeps up)  
- fake/teacher: denoise error of fake score vs teacher on generator samples (<1 means the fake score has fitted the generator distribution)  
- p_real / p_fake, loss_d / loss_g_gan: discriminator outputs and GAN losses  

![dmd2_loss_curve](figs/dmd2_loss_curve.png)
#### 3.8 1-step DDIM / RF / CT / CD / MeanFlow / DMD2   
![1step_cmp](figs/1step_gen_cmp.png)  
#### 3.9 LoRA finetune
four settings, r=1(0.33%), r=4(1.34%), r=16(5.36%), full(99.49%)  
loss curve  
![lora_loss_curve](figs/rf_lora_loss_cmp.png)
samples generated during LoRA finetune. synthetic outlined MNIST as LoRA finetune dataset     
![lora_progress](figs/lora_finetune_progress.png)
comparison between LoRA finetune and full parameter finetune  
![lora_full_cmp](figs/lora_full_cmp.png)  

### 4. pitfalls
- **MeanFlow time scale**: feeding `t * 1000` (as in RF) into the sinusoidal embedding amplifies du/dt by ~1000x (∂u/∂(t-r) measured 14–43 vs |v|≈1.4), the JVP target is dominated by it and training diverges (blurry, wrong-class 1-step samples). Feed t and t-r in [0, 1] as the official implementation does.
- **MeanFlow JVP memory**: default SDPA kernels do not support forward-mode AD, use the MATH backend for the JVP. Computing the JVP with autograd also stores the tangent graph (~50GB at batch 256); since the target is stop-grad, compute the JVP under `no_grad` and get u from a separate normal forward (~10GB).
- **CT time distribution**: RF's logit-normal(0, 1) puts only ~0.16% of samples at t ≥ 0.95; CT collapsed around epoch 40 (after the N 40→80 switch) in repeated runs. iCT's lognormal (P_mean=-1.1, P_std=2.0, ~2.2% at t ≥ 0.95) did not collapse.
- **DDPM few-step sampling grid**: `linspace(999, 0, steps)` wastes the last network call at t=0, so 2 steps ≈ 1 step. Use the trailing grid `linspace(999, 0, steps + 1)[:-1]`. At 1–2 steps DPM++(2M) and UniPC are identical to DDIM.
- **eps-prediction at 1 step**: x0 = (x_t - √(1-ᾱ)·ε̂) / √ᾱ with √ᾱ_999 ≈ 0.0064 amplifies the ε error ~156x, so 1-step DDIM with eps-prediction is pure noise while x0 / v give blurry means.

### 5. citations    
DDPM [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
Rectified Flow [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)  
DDIM [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)  
DPM-Solver-2 [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)  
DPM++ [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095)  
UniPC [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2302.04867)  
Heun [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)  
v-prediction [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)  
cosine schedule [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)  
trailing, zero terminal SNR [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)  
CFG [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)  
logit-normal, timestep shift [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)  
Consistency Models [Consistency Models](https://arxiv.org/abs/2303.01469), [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189)  
MeanFlow [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)  
DMD [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828)  
DMD2 [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867)  
DiT [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)  
LoRA finetune [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)  
min-SNR-𝛄 [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)  