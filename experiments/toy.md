I read the proposal. The architectural center of gravity is not “a diffusion model” so much as a **decoder (g_\theta)** plus **one time-conditioned encoder (f_\phi(x,t))**, with the plain encoder defined as the (t=0) case, so your roundtrip denoiser is (R_t(y_t)=f_\phi(g_\theta(y_t),t)) and its job is to predict the denoised latent (E[y_0\mid y_t]). That matches the proposal’s framing and the Tweedie/de Bruijn story in your notes: turn score estimation into regression, and learn the posterior mean rather than the score directly.

My strong recommendation is: **do not invent a new time-conditioning scheme**. Use exactly what diffusion models use.

### How to parameterize (t)

The standard pattern is:

1. take a scalar timestep/noise level,
2. turn it into a sinusoidal or Fourier embedding,
3. pass that through a small MLP,
4. inject the result into every residual block, either by simple addition or by FiLM/scale-shift modulation.

That is exactly how the common diffusion U-Net implementations are built. In Diffusers, `UNet2DModel` takes a timestep as `torch.Tensor`, `float`, or `int`, exposes `time_embedding_type` (`positional` or `fourier`), and exposes `resnet_time_scale_shift` (`default` or `scale_shift`). The annotated diffusion implementation from Hugging Face uses sinusoidal embeddings, an MLP, and then uses the time embedding inside each ResNet block via scale/shift. Guided-diffusion’s timestep helper explicitly says the timesteps may be fractional, so continuous-time conditioning is normal rather than exotic. ([Hugging Face][1])

For **your** path (y_t=(1-t)y_0+t\varepsilon), I would just feed **raw normalized (t\in[0,1])** on day one. You do not need log-SNR gymnastics yet, because in your setup (t) already *is* the noise scale. If you later move to VP/VE schedules, then switching the conditioning scalar from raw (t) to (\sigma) or log-SNR becomes more attractive. Also, because your exact MLE weighting blows up near (t\to 0), I would initially sample (t) from a truncated interval like ([0.02,1]) and start with your milder FM-style weighting before switching on the exact (\alpha_t=(1-t)/t^3). That is very much in line with your own proposal’s alternate weighting and your note that the de Bruijn decomposition lets you trade off bias and variance by truncating the hard low-noise regime.

### Which toolkit stack to use

If I had to pick one practical stack, I would use **plain PyTorch for the custom losses**, **Diffusers for reusable time-conditioned blocks / U-Nets**, and optionally **TorchCFM or Meta’s `flow_matching`** as reference implementations for probability paths and toy baselines.

Diffusers is the best reusable backbone source here. Their unconditional training guide already gives you a working `UNet2DModel` + scheduler + AdamW + Accelerate skeleton, so you can steal the model and training plumbing without adopting their loss. Their schedulers also already expose different prediction targets; `DDPMScheduler` supports `epsilon`, `sample`, and `v_prediction`, and newer `DPMSolverMultistepScheduler` also supports `flow_prediction`. ([Hugging Face][2])

For flow-matching references, Meta’s `flow_matching` package is now an official PyTorch library with synthetic and image training examples, and `torchcfm` remains very handy because it explicitly ships flow-matching variants, 2D examples, and an unconditional MNIST notebook. ([GitHub][3])

For the autoencoder part, the official `pytorch/examples` repo is a good seed because it is explicitly meant to be “curated, short, few/no dependencies” and it includes a VAE example. Even if you do not keep the stochastic VAE loss, it is a nice low-friction source for the encoder/decoder skeleton. ([GitHub][4])

### Toy Gaussian: what I would actually build

For the toy stage, I would **not** learn the encoder/decoder first. Your proposal is trying to show that latent denoising avoids the ambient-dimension/KDE pain point, so the cleanest toy is:

* latent (y_0 \in \mathbb R^2), sampled from a Gaussian mixture or another analytically tractable toy;
* observed data (x = Ay) or (x = Ay + \xi), where (A) is a fixed random orthonormal embedding into (\mathbb R^d);
* fixed decoder (g(y)=Ay);
* fixed encoder (f(x)=A^\dagger x);
* learn only the time-conditioned denoiser (f_\phi(x,t)) or equivalently (R_t).

That removes the autoencoder as a confound and directly tests the proposal’s stated claim about high-dimensional density-estimation variance. Then you can sweep (d=2,16,64,256) and compare x-space KDE/drifting baselines against the latent regression route. That experiment is almost tailor-made for the proposal’s first claim.

Architecture-wise, use a **small residual MLP**, not a U-Net. Something like 4 residual blocks, width 128 or 256, SiLU activations, with a 64-d sinusoidal time embedding expanded by a 2-layer MLP, is plenty. For the absolute smallest starting point, TorchCFM’s 2D examples note that a **3x64 MLP** is enough for those toy flows. ([GitHub][5])

A good minimal module is:

* `time_mlp`: sinusoidal/Fourier 64 -> Linear 128 -> SiLU -> Linear 128
* `block`: LayerNorm -> Linear -> SiLU -> Linear, with time injected by concat or FiLM
* `head`: hidden -> latent dim

For MLPs, concatenating `[h, t_emb]` at every block is the least-effort version. For conv models, use scale-shift.

### MNIST: two viable starting points

For MNIST, there are really two good choices.

**Absolute least effort:** use a flattened MLP autoencoder.
Encoder: `784 -> 400 -> 64`, decoder: `64 -> 400 -> 784`.
Then make the encoder time-conditioned in the same way as the toy MLP, and define your plain encoder as `f(x, 0)`. This is the fastest way to test whether the objective works at all, and you can bootstrap it from the official PyTorch VAE example skeleton. ([GitHub][4])

**Better low-risk image model:** use a tiny convolutional autoencoder.
A very sane setup is:

* encoder: `1x28x28 -> 32x14x14 -> 64x7x7 -> latent`
* latent: either a 64-d vector **or**, better, a **4x7x7 spatial latent map**
* decoder: mirror it back to `1x28x28`
* time-conditioned encoder: same conv backbone, but each residual block gets the diffusion-style time embedding via scale-shift

The reason I like the **4x7x7 latent map** is that it makes your later baselines much easier: you can compare your roundtrip method against a plain latent diffusion / latent flow-matching baseline using an off-the-shelf `UNet2DModel` directly on the latent tensor. That is a very strong ablation because it tests whether your gain comes from the **roundtrip/data-measure trick**, not just from “doing things in latent space.” Diffusers’ `UNet2DModel` is a natural fit here, and its input size only needs to be divisible by (2^{L-1}), so 28 works fine for a 3-level U-Net. ([Hugging Face][1])

If you do use a tiny U-Net anywhere in MNIST, I would start with something in this ballpark:

* `sample_size=28`
* `in_channels=1`, `out_channels=1` for image-space baselines, or `4/4` for a 4x7x7 latent-map baseline
* `block_out_channels=(32, 64, 64)`
* `layers_per_block=2`
* `time_embedding_type="positional"`
* `resnet_time_scale_shift="scale_shift"`
* zero or one attention block at the lowest resolution

I would **not** use a Transformer/DiT on MNIST. It adds code and compute without helping you prove the point.

### One architectural simplification that matters

Do **not** implement separate (f_\phi) and (f_\phi^t) networks. Implement **one** network `encoder(x, t)` and define the plain encoder as `encoder(x, 0)`. That keeps the code small, enforces the proposal’s intended relation (f_\phi = f_\phi^0), and makes it easy to pretrain reconstruction/cycle losses and then turn on denoising losses without changing the model class.

### What target the network should predict

Because your proposal is explicitly estimating the posterior mean (E[y_0\mid y_t]), I would keep the model head as an **x0/sample-prediction head**, not an epsilon-prediction head. In diffusion-library language, your objective is closer to `prediction_type="sample"` than to the default epsilon objective. Only switch to velocity/flow targets later if you decide you want sampler compatibility with flow-prediction solvers rather than the clean Tweedie interpretation.   ([Hugging Face][6])

So the short answer is:

* **Toy Gaussian:** fixed linear encoder/decoder + time-conditioned residual MLP.
* **MNIST first pass:** flattened MLP autoencoder if you want maximum speed.
* **MNIST recommended:** tiny conv autoencoder, with a single time-conditioned encoder using diffusion-style sinusoidal time embeddings and scale-shift residual blocks.
* **Steal (t)-conditioning from diffusion models exactly as-is.**
* **Keep your novelty in the loss / roundtrip construction, not in the backbone.**

The cleanest first stack is: **PyTorch + Diffusers blocks, TorchCFM for toy references, and one shared `encoder(x,t)` with `t=0` as the ordinary encoder.**

[1]: https://huggingface.co/docs/diffusers/main/en/api/models "https://huggingface.co/docs/diffusers/main/en/api/models"
[2]: https://huggingface.co/docs/diffusers/training/unconditional_training "https://huggingface.co/docs/diffusers/training/unconditional_training"
[3]: https://github.com/facebookresearch/flow_matching "https://github.com/facebookresearch/flow_matching"
[4]: https://github.com/pytorch/examples "https://github.com/pytorch/examples"
[5]: https://github.com/atong01/conditional-flow-matching "https://github.com/atong01/conditional-flow-matching"
[6]: https://huggingface.co/docs/diffusers/v0.22.3/api/schedulers/ddpm "https://huggingface.co/docs/diffusers/v0.22.3/api/schedulers/ddpm"
