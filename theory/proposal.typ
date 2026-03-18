#let hyperlink-blue = rgb("#1f4b99")
#let blue(body) = text(fill: hyperlink-blue, body)
#show link: it => underline(
  stroke: hyperlink-blue,
  text(fill: hyperlink-blue, it.body),
)

= Drifting research proposal
Nicholas (Xingjian) Lyu. Mar 15, 2026
\

*TLDR*: we propose a latent drifting protocol which #blue[_sidesteps density estimation variance in high-dimensions_] and #blue[_performs score matching_] under the correct data measure #footnote[Score matching under the model distribution minimizes the mode-seeking #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#ex-otto-reverse-kl")[reverse KL], instead of the proper MLE objective.]. The protocol admits a clean MLE interpretation.

The main mathematical tools are #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#thm-tweedie")[Tweedie's formula] and #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#prp-fm")[de Bruijn's identity]. In my opinion, they are the mathematical powerhouse behind the stability and dimension-scalability of flow matching models. We couple these tools with one-step generation by using an encoder-decoder construction #footnote[equivalently, generator-critic; there are many perspectives here]. This construction also addresses the reverse-KL sampling challenge for many one-step generation models, including drifting, using a roundtrip trick.

== Interpretation of drifting; challenges

We have shown in #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#gaussian-kernel-smoothing-implements-reverse-kl")[these notes] that drifting implements Wasserstein gradient descent on the reverse-KL divergence $D(rho_"model" || rho_"data")$ using kernel density estimates (KDE) of the model and data densities. The drifting field at each sample is the Gaussian kernel-estimate of score difference between data and model distributions at that point#footnote[#link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#ex-otto-reverse-kl")[Applying Otto's theorem to reverse KL] shows that the score-difference field on sample space is the Wasserstein gradient of reverse KL.]. This perspective suggests the following challenges with existing drifting methods:

1. *Gaussian KDE variance scales with data dimension*: this is one possible explanation for why ImageNet drifting does not work without a good pretrained encoder.
2. *KL objective is stiff*: KL blows up when supports don't overlap. When data and model distributions don't overlap, drifting fails because the fundamental Wasserstein-functional objective blows up, _even if density estimates were perfect_. This is another possible explanation for why drifting struggles to natively scale to high dimensions.
3. *The reverse-KL sampling problem*: it's well-known in generative modeling and RL literature  that optimizing the reverse KL $D(rho_"model" || rho_"data")$ is prone to mode collapse and low diversity. However, optimizing the proper forward-KL objective is hard because it requires sampling the model w.r.t. the _data distribution_ $EE_"data"$, while reverse-KL samples the model naturally under $EE_"model"$. Typical one-shot models cannot sample from $EE_"data"$ _because we can't tell which latent could have generated a data sample_.


== Escape hatches

I did #link("http://localhost:4321/blogs/machine-learning/ot-generative-3-diffusion/")[some learning on diffusion / flow-matching] to see how these problems are addressed in the FM paradigm. The main takeaways are:

1. _Reduce density (score) estimation to regression_: #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#thm-tweedie")[Tweedie's formula] reduces score estimation to denoising #footnote[$x \| z tilde.op cal(N)(alpha z, Sigma)$]. The difficulty of parametric regression scales with data size and the structural (manifold) dimension of the problem instead of the data dimension.
$
  alpha EE[z \| x] = x + sigma^2 nabla log rho
$
2. _Decompose KL along the noise spectrum into score matching_: by decomposing KL into a score matching integral, we can make powerful bias-variance tradeoffs to e.g. truncate the divergent $t->0$ component, reweight score matching losses, and importance sample, etc#footnote[The de Bruijn integral we presented assumes the (dominantly used) flow matching process, which corresponds to optimal-transport of the independent data-noise coupling.]

== High-level proposal

Our proposal is motivated by applying the escape hatches above to drifting with MLE in mind. From first principles, properly #footnote[we could consider importance sampling, but it introduces high variance and dependence upon explicit density estimation of $rho_"model" slash rho_"data"$.] solving the reverse-KL sampling problem requires us to know _what latent would have generated a data sample_; this calls for an inverse $f_phi$ to the generator $g_theta$. We naturally propose an encoder-decoder architecture
$
  "latent" stretch(harpoons.rtlb)^(g_theta)_(f_phi) "model sample", quad g_theta compose f_phi = f_phi compose g_theta = "Id  on relevant supports"
$
We know that the final data latent we target is the initial noise $nu = cal(N)(0, 1)$; this is a closed-form target, so it appears natural to score-match in latent space. We use #blue[$rho^+$ for the data distribution, $rho^- = g_(theta\#) (nu)$ for the model distribution, and $x tilde.op rho^plus.minus$ for samples]. We also write #blue[$sigma^plus.minus = f_(phi\#) (rho^plus.minus)$ for the latent distributions, and $y tilde.op sigma^plus.minus$ for latents.]
$
  "latent" nu arrow.r^(g_theta) "sample" rho^- arrow.r^(f_phi) sigma^-, quad "true data" rho^+ arrow.r^(f_phi) sigma^+
$
We know that $f_phi compose g_theta = "Id" ==> sigma^- approx nu$, and we define the noised data latents $sigma^+_t$, which linearly interpolate to i.i.d. Gaussian with $t in [0, 1]$, such that (using intuitive expressions):
$
  sigma^plus.minus_t = (1-t) sigma^plus.minus + t cal(N)(0, I)
$
This is the canonical noise process in flow matching. Since $sigma^- approx nu =cal(N)(0, I)$ the noised distributions are analytic $sigma^-_t = cal(N)(0, (t^2 + overline(t)^2) I)$

Tweedie reduces score estimation to estimating the denoised data latent $EE_plus [y_0 \| y_t]$. *We parameterize the latent data-denoiser $(y_t, t) mapsto EE_+ [y_0 \| y_t]$ by a round-trip*
$
  (f_phi^t compose g_theta)(y_t) approx EE_plus [y_0 \| y_t]
$
Note that score-matching in latent space with $f_phi compose g_theta = "Id"$ has bought us a free, analytic model latent distribution (i.i.d. Gaussian), so we only need to learn the data latents.
The encoder $f_phi^(t)$ now accepts additional parameterization; #blue[we continue to denote $f_phi = f_phi^0$]. The denoising objective at $t=0$ coincides with $f_phi compose g_theta = "Id"$ continuously.

=== High-level objectives <high-level-objectives>
The detailed losses which we'll proceed to developing all serve the following objectives:

1. Encoder-decoder bijectivity: $g_theta compose f_phi = f_phi compose g_theta = "Id"$.
2. Round-trip estimates of latent data scores via denoising: $(f_phi^t compose g_theta )(y_t) approx EE_plus [y_0 \| y_t]$
3. Minimize mismatch between the estimated data latent score and the analytic Gaussian score:
$
  EE_(y tilde.op sigma^+_t) ||(f_phi^t compose g_theta )(y_t) - EE_-^*[y_0 | y_t]||^2
$

#pagebreak()

== Theory

The theory below actually motivated our design. Recalling definitions of $rho^plus.minus, sigma^plus.minus$, apply the #link("https://nlyu1.github.io/classical-info-theory/kullback-leibler-divergence.html#chain-rule-dpi")[data processing inequality (DPI)] to expand sample-space KL in latent-space KL with precise residual:
$
  D(rho^+ || rho^-) = D(sigma^+ || sigma^-) + EE_(y tilde.op sigma^+) D(P_(rho^+ | y) || P_(rho^-|y))
$
The first term measures latent-space divergence, and the second-term quantifies how lossy is the encoder channel $f_phi$. Applying the #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#flow-matching-in-practice")[standard flow matching formulas] (de Bruijn + Tweedie) yields the latent KL expansion
$
  D(sigma^+ || sigma^-) = EE_(t tilde.op "Unif"[0, 1]) [alpha_t EE_(y_t tilde.op sigma^+_t ) ||EE_+[y_0^+ | y_t] - EE_-[y_0^- | y_t]||^2]
$
Where $alpha_t = (1-t) slash t^3$ is the noise schedule that corresponds to MLE. We'll now translate the forward-KL components into operational losses that can be sampled. #blue[Define $R^t_(phi theta) = f_phi^t compose g_theta$ with $t$ defaulting to $0$, $"sg"(v)$ for stopgrad / detach, and $overline(t) := 1-t$]. There are two cycle losses, one denoising loss, one score matching loss, and one reconstruction loss:
$
  cal(L)_"denoise" (phi, theta) & := EE_(t tilde.op "Unif"[0, 1], x tilde.op rho^+, epsilon tilde.op cal(N)(0, I)) [
    alpha_t ||y_0^+ - R^t_(phi theta) (y_t^+)||^2
  ] \
  y_0^+ & := "stopgrad"(f_phi (x)), quad y_t^+ := overline(t) y_0^+ + t epsilon \
  cal(L)_"score" (theta) &:= EE_(t tilde.op "Unif"[0, 1], x tilde.op rho^+, epsilon tilde.op cal(N)(0, I))||tilde(y) - "sg"(V(tilde(y), epsilon, t) + tilde(y))||^2 \
  V (y, epsilon, t) & := -alpha_t dot nabla_y ||R^t_(phi theta)(y_t) - EE^*_-[y_0 | y_t ]||^2 \
  y_t & := overline(t) y + t epsilon, quad tilde(y) := R_(phi theta)("sg"(f_phi (x))), quad EE_-^*[y_0 | y_t] := (overline(t) y_t) / (overline(t)^2 + t^2)\
  cal(L)_"cycle"^- (theta, phi) & := EE_(y tilde.op nu)||R_(phi theta)(y) - y||^2 \
  cal(L)_"cycle"^+ (theta, phi) & := EE_(y tilde.op sigma^+)||R_(phi theta)(y) - y||^2 \
  cal(L)_"dpi" (theta, phi) & := EE_(x tilde.op rho^+)||(g_theta compose f_phi) (x) - x||^2
$
Use $alpha_t = overline(t) slash t^3$ for exact MLE, and $alpha_t = 1 slash t^2$ for flow-matching's uniform-weighted velocity matching weighting.

=== Score matching $=>$ encoder denoising + decoder drifting
Define the score matching loss at time $t$ and latent $y_t$:
$
  L(y_t, t) := EE_(y_t tilde.op sigma_t^+) ||EE_+[y_0^+ | y_t] - EE_-[y_0^- | y_t]||^2
$
To reduce clutter, suppress dependence on $y_t$ and define the true posterior (denoised) latents
$
  y_+^* := EE_+[y_0^+ | y_t], quad y_-^* := EE_-[y_0^- | y_t]
$
and the round-trip approximations; note that our "estimate" of the model latent denoiser is given by the analytic Gaussian denoiser $EE_-^*$:
$
  hat(y_+^*) := (f_phi^t compose g_theta)(y_t), quad hat(y_-^*) := EE_-^*[y_0 | y_t ] = (overline(t) y_t) / (overline(t)^2 + t^2)
$
Critically note that #blue[$y_t$ needs to be sampled from the noised _data_ latent $sigma_t^+$]:
$
  L(y_t, t) = EE_(y_t tilde.op sigma_t^+) ||y_+^* - y_-^*||^2
$
Using standard inequality $||a+b+c||^2slash 3 <= ||a||^2 + ||b||^2 + ||c||^2$, we obtain
$
  1/3 L(y_t, t)
  <= EE_(y_t tilde.op sigma_t^+) [
    ||y_+^* - hat(y_+^*)||^2
    + ||y_-^* - hat(y_-^*)||^2
    + ||hat(y_-^*) - hat(y_+^*)||^2
  ]
$
This decomposes the target score-matching objective into _data-denoising fidelity_, _model-denoising fidelity_, and _estimated latent score mismatch_, respectively.

==== Data-latent denoising

It's fairly straightforward to design the data denoising loss. This is an encoder-side objective, so we want to minimize it w.r.t. $phi$ while treating $theta$ as fixed.
$
  cal(L)_"denoise"^+ := EE_(t) EE_((y_0, y_t) tilde.op sigma_t^+) [alpha_t ||y_0 - R^t_(phi theta) (y_t)||^2]
$
In practice, the encoder's job is only possible when the decoder $g_theta$ provides invertible support, so we also optimize $theta$ (but with a lower learning rate). To practically sample from this loss, write the expectation using the given data distribution $rho^+$:
$
  cal(L)_"denoise"^+
  &:= EE_(t) EE_(x tilde.op rho^+ \ epsilon tilde.op cal(N)(0, I)) [alpha_t ||y_0 - R^t_(phi theta) (y_t)||^2]\
  y_t &:= overline(t) y_0 + t epsilon, quad y_0 := "stopgrad"(f_phi (x))
$
Note the stopgrad application of $f_phi$ as well: we're just using $f_phi$ to sample from the pushforward.

==== Estimated score mismatch

The score matching loss is $EE_(y_t tilde.op sigma_t^+) ||hat(y_-^*) - hat(y_+^*)||^2$ with analytic expression
$
  D(sigma^+ || sigma^-) = EE_(t, y_t tilde.op sigma^+_t) [alpha_t||R^t_(phi theta)(y_t) - EE_-^*[y_0 | y_t]||^2]
$
We want to minimize this loss w.r.t. $theta$, while holding the round-trip score estimator $R^t_(phi theta)$ fixed.

One big problem with the loss above is that #blue[$y_t tilde.op sigma_t^+$ has no generator dependency]. This problem generally plagues one-shot generation models: the forward KL (or score matching) requires optimizing $theta$ under the true data measure, but data don't have $theta$-dependency.

We introduce the *roundtrip trick*: given model latent $y=f_phi (x) tilde.op sigma^+$, it naively doesn't have $theta$-dependence. However, we can impose the reconstruction $g_theta compose f_phi approx "Id"$ over $rho^+$ to infer which latent $y = f_phi (x)$ could have approximately produced the data sample $x$, then the round-trip latent
$
  tilde(y) = R_(phi theta) (y)
$
has proper generator $theta$ dependency. To _ensure roundtrip fidelity_, we need
$
  cal(L)_"cycle"^+ (theta, phi) & := EE_(y tilde.op sigma^+)||R_(phi theta)(y) - y||^2
$
To produce the _proper latent estimate for data sample $x$_,  we need
$
  cal(L)_"dpi" (theta, phi) & := EE_(x tilde.op rho^+)||(g_theta compose f_phi) (x) - x||^2
$
Using this trick, the surrogate loss is
$
  D(sigma^+ || sigma^-) approx cal(L)_"score" (theta) = EE_(t, y_t tilde.op sigma^+_t)[alpha_t||R^t_(phi theta) (tilde(y)_t) - EE_-^*[y_0 | tilde(y)_t]||^2]
$
Optimizing this loss with frozen $R^t_(phi theta)$ with respect to $theta$-dependency in $tilde(y)_t$ is equivalent to drifting.

Another interpretation of drifting is as follows; in the ideal limit $R^t_(phi theta) = EE^*_+[y_0 | y_t]$, consider the potential of the (time, noise)-amortized drifting field
$
  EE_(t, epsilon)V (y, epsilon, t) & approx -nabla_y Phi(y) \
  Phi(y) & = EE_(t, epsilon) [alpha_t||EE^*_+[y_0 | y_t] - EE^*_-[y_0 | y_t ]||^2] \
  y_t & = overline(t)y + t epsilon, quad EE_-^*[y_0 | y_t] := (overline(t) y_t) / (overline(t)^2 + t^2) \
$
The expectation of the drifting potential over the data latent distribution $y tilde.op sigma^+$ is $D(sigma^+ || sigma^-)$ via de Bruijn and Tweedie.
$
  D(sigma^+ || sigma^-) = EE_(y tilde.op sigma^+) Phi(y) = integral Phi(y) dif sigma^+(y)
$
The first variation of this functional w.r.t. $sigma^+(y)$ is $Phi(y)$; applying Otto's formula yields the Wasserstein gradient $nabla Phi(y)$. In this sense, we reparameterize $sigma^+$ using the roundtrip trick and use drifting to minimize $D(sigma^+ || sigma^-)$.

==== Model-latent denoising

The model-latent denoising objective is
$
  cal(L)_"denoise"^- := EE_(t) [alpha_t dot EE_(y_t tilde.op sigma_t^-) ||EE_-[y_0 | y_t ] - EE_-^*[y_0 | y_t]||^2]
$
with an analytic Gaussian denoiser
$
  EE_-^*[y_0 | y_t] = (overline(t)) / (overline(t)^2 + t^2) y_t
$
We surrogate-optimize this loss by guiding $sigma^- -> nu <== R_(phi theta) approx "Id"$ over $nu$. This yields
$
  cal(L)_"cycle"^- (theta, phi) & := EE_(y tilde.op nu)||R_(phi theta)(y) - y||^2
$

=== DPI bound $=>$ data reconstruction loss

The DPI term $EE_(y tilde.op sigma^+) D(P_(rho^+ | y) || P_(rho^-|y))$ is surrogate-optimized by the negative log-likelihood of a Gaussian decoder channel model, which reduces to a squared reconstruction loss:
$
  cal(L)_"dpi" (theta, phi) = EE_(x tilde.op rho^+) ||(g_theta compose f_phi) (x) - x||^2
$
After observing latent $y$, model the posterior over $rho^-$ as a Gaussian centered at decoder output $g_theta (y)$ with variance $beta$:
$
  hat(rho) (x | y) = cal(N)(x; g_theta (y), beta I)
$
Note that because the encoder is deterministic, $EE_(y tilde.op sigma^+, x tilde.op rho^+ | y) = EE_(x tilde.op rho^+)$; expanding the DPI term yields the reconstruction loss:
$
  EE_(y tilde.op sigma^+) D(P_(rho^+|y) || P_(rho^-|y)) & = EE_(y tilde.op sigma^+, x tilde.op rho^+|y) [
    log rho^+(x|y) - log hat(rho)(x|y)
  ] \
  & = -EE_(x tilde.op rho^+) log hat(rho) (x | f_phi (x)) - h(X^+ | Y) \
  & <= EE_(x tilde.op rho^+) 1/(2 beta) ||x - g_theta (f_phi (x))||^2 + C
  = 1/(2 beta) cal(L)_"dpi" + C
$
