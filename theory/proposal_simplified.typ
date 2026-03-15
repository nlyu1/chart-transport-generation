#let hyperlink-blue = rgb("#1f4b99")
#let blue(body) = text(fill: hyperlink-blue, body)
#show link: it => underline(
  stroke: hyperlink-blue,
  text(fill: hyperlink-blue, it.body),
)

= Drifting research proposal
Nicholas (Xingjian) Lyu. Mar 8, 2026
\

*TLDR*: we propose a latent drifting protocol which #blue[_sidesteps density estimation variance in high-dimensions_] and #blue[_performs score matching_] under the correct data measure #footnote[Score matching under the model distribution minimizes the mode-seeking #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#ex-otto-reverse-kl")[reverse KL], instead of the proper MLE objective.]. The protocol admits a clean MLE interpretation.

The main mathematical tools are #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#thm-tweedie")[Tweedie's formula] and #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#prp-fm")[de Bruijn's identity]. In my opinion, they are the mathematical powerhorse behind the stability and dimension-scalability of flow matching models. We couple these tools with one-step generation by using a encoder-decoder construction #footnote[equivalently, generator-critic; there're many persepectives here]. This construction also addresses the reverse-KL sampling challenge for many one-step generation models, including drifting, using a roundtrip trick.

== Interpretation of drifting; challenges

We have shown in #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#gaussian-kernel-smoothing-implements-reverse-kl")[these notes] that drifting implements Wasserstein gradient descent on the reverse-KL divergence $D(rho_"model" || rho_"data")$ using kernel density estimates (KDE) of the model and data densities. The drifting field at each sample is the Gaussian kernel-estimate of score difference between data and model distributions at that point#footnote[#link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#ex-otto-reverse-kl")[Applying Otto's theorem to reverse KL] shows that the score-difference field on sample space is the Wasserstein gradient of reverse KL.]. This perspective suggests the following challenges with existing drifting methods:

1. *Gaussian KDE variance scales with data dimension*: this is one possible explanation for why ImageNet drifting does not work without a good pretrained encoder.
2. *KL objective is stiff*: KL blows up when supports don't overlap. When data and model distributions don't overlap, drifting fails because the fundamental Wasserstein-functional objective blows up, _even if density estimates were perfect_. This is another possible explanation for why drifting struggles to natively scale to high dimensions.
3. *The reverse-KL sampling problem*: it's well-known in generative modeling and RL literature  that optimizing the reverse KL $D(rho_"model" || rho_"data")$ is prone to mode collapse and low diversity. However, optimizing the proper forward-KL objective is hard because it requires sampling the model w.r.t. the _data distribution_ $EE_"data"$, while reverse-KL samples the model naturally under $EE_"model"$. Typical one-shot models cannot sample from $EE_"data"$ _because we can't tell which latent could have generated a data sample_.


== Escape hatches

I did #link("http://localhost:4321/blogs/machine-learning/ot-generative-3-diffusion/")[some learning on diffusion / flow-matching] to see how these problems are addressed in the FM paradigm. The main takeaways are:

1. _Reduce density (score) estimation to regression_: #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#thm-tweedie")[Tweedie's formula] reduces score estimation into denoising #footnote[$x \| z tilde.op cal(N)(alpha z, Sigma)$]. The difficulty of parametric regression scales with data size and the structural (manifold) dimension of the problem instead of the data dimension.
$
  alpha EE[z \| x] = x + sigma^2 nabla log rho
$
2. _Decompose KL along the noise spectrum into score matching_: by decomposing KL into a score matching integral, we can make powerful bias-variance tradeoffs to e.g. truncate the divergent $t->0$ component, reweight score matching losses, and importance sample, etc#footnote[The de Bruijn integral we presented assumes the (dominantly used) flow matching process, which corresponds to optimal-transport of the independent data-noise coupling.]

== High-level proposal

Our proposal is motivated to apply the escape hatches above to drifting with MLE in mind. From first principles, solving the reverse-KL sampling problem properly #footnote[we could consider importance sampling, but it introduces high variance and dependence upon explicit density estimation of $rho_"model" slash rho_"data"$.] requires us to know _what latent would have generated a data sample_; this calls for an inverse $f_phi$ to the generator $g_theta$. We naturally propose an encoder-decoder architecture
$
  "latent" stretch(harpoons.rtlb)^(g_theta)_(f_phi) "model sample", quad g_theta compose f_phi = f_phi compose g_theta = "Id  on relevant supports"
$
We know that the final data latent we target the is initial noise $nu = cal(N)(0, 1)$; this is a closed-form target, so it appears natural to score-match in latent space. We use #blue[$rho^+$ for the data distribution, $rho^- = g_(theta\#) (nu)$ for the model distribution, and $x tilde.op rho^plus.minus$ for samples]. We also write #blue[$sigma^plus.minus = f_(phi\#) (rho^plus.minus)$ for the latent distributions, and $y tilde.op sigma^plus.minus$ for latents.]
$
  "latent" nu arrow.r^(g_theta) "sample" rho^- arrow.r^(f_phi) sigma^-, quad "true data" rho^+ arrow.r^(f_phi) sigma^+
$
We know that $f_phi compose g_theta = "Id" ==> sigma^- = nu$, we also define the noised latents $sigma^plus.minus_t$ which linearly interpolates to i.i.d. Gaussian with $t in [0, 1]$. such that (using intuitive expressions):
$
  sigma^plus.minus_t = (1-t) sigma^plus.minus + t cal(N)(0, I)
$
This is the canonical noise process in flow matching. Since $sigma^-=nu =cal(N)(0, I)$ the noised distributions are stationary $sigma^-_(forall t) = sigma^-=nu$

Tweedie reduces score estimation to estimating the denoised data latent $EE_plus [y_0 \| y_t]$. *We parameterize the latent data-denoiser $(y_t, t) mapsto EE_+ [y_0 \| y_t]$ by a round-trip*
$
  (f_phi^t compose g_theta)(y_t) approx EE_plus [y_0 \| y_t]
$
Note that score-matching in latent space with $f_phi compose g_theta = "Id"$ have bought us a free, analytic model latent distribution (i.i.d. Gaussian), so we only need to learn the data latents.
The encoder $f_phi^(t)$ now accepts additional parameterization; #blue[we continue to denote $f_phi = f_phi^0$]. Note that the denoising objective at $t=0$ coincides with $f_phi compose g_theta = "Id"$ continuously.

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
Where $alpha_t = (1-t) slash t^3$ is a specific noise schedule. We'll now translate the forward-KL components into operational losses that can be sampled. Here's the complete catalog which transcribes the #link(<high-level-objectives>)[high-level objectives]. There are two denoising losses, one decoder-side score loss, and two cycle losses:
$
  cal(L)_"denoise"^+ (phi) & := EE_(t tilde.op "Unif"[0, 1], x tilde.op rho^+, epsilon tilde.op cal(N)(0, I)) [
    alpha_t ||y_0^+ - (f_phi^t compose g_theta)(y_t^+)||^2
  ] \
  y_0^+ & := "stopgrad"(f_phi (x)), quad y_t^+ := (1-t) y_0^+ + t epsilon \
  cal(L)_"denoise"^- (phi) & := EE_(t tilde.op "Unif"[0, 1], z tilde.op nu, epsilon tilde.op cal(N)(0, I)) [
    alpha_t ||y_0^- - (f_phi^(-t) compose g_theta)(y_t^-)||^2
  ] \
  y_0^- & := "stopgrad"((f_phi compose g_theta)(z)), quad y_t^- := (1-t) y_0^- + t epsilon \
  cal(L)_"score" (theta) & := EE_(y tilde.op sigma^+, t tilde.op "Unif"[0, 1], epsilon tilde.op cal(N)(0, I)) [
    ||"stopgrad"[V (tilde(y), epsilon, t) + tilde(y)] - tilde(y)||^2
  ] \
  V (y, epsilon, t) & := -alpha_t dot nabla_y ||overline((f_phi^t compose g_theta))(y_t) - overline((f_phi^(-t) compose g_theta))(y_t)||^2 \
  y_t & := (1-t) y + t epsilon, quad tilde(y) := (f_phi compose g_theta)(y) \
  cal(L)_"roundtrip" (theta, phi) & := EE_(y tilde.op sigma^+) ||(f_phi compose g_theta)(y) - y||^2 \
  cal(L)_"reconstruct" (theta, phi) & := EE_(x tilde.op rho^+) ||g_theta (f_phi (x)) - x||^2
$
Where $alpha_t in RR$ is the noise schedule and overline denotes function application without gradient capture. Use $alpha_t = (1-t) slash t^3$ for exact MLE, and $alpha_t = 1 slash t^2$ for flow-matching's uniform-weighted velocity matching weighting.

=== Score matching $=>$ encoder denoising + decoder drifting
Define the score matching loss at time $t$ and latent $y_t$:
$
  L(y_t, t) := EE_(y_t tilde.op sigma_t^+) ||EE_+[y_0^+ | y_t] - EE_-[y_0^- | y_t]||^2
$
To reduce clutter, suppress dependence on $y_t$ and define the true posterior (denoised) latents
$
  y_+^* := EE_+[y_0^+ | y_t], quad y_-^* := EE_-[y_0^- | y_t]
$
and the round-trip approximations; note that our "estimate" of the model latent denoiser is given by the analytic i.i.d. Gaussian denoiser $EE_-^*$ (we'll derive closed-form expression soon):
$
  hat(y_+^*) := (f_phi^t compose g_theta)(y_t), quad hat(y_-^*) := EE_-^*[y_0 | y_t ]
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

It's fairly straightforward to design the data denoising loss. This is an encoder-side objective, so we minimize it w.r.t. $phi$ while treating $theta$ as fixed.
$
  cal(L)_"denoise"^+ := EE_(t) [alpha_t dot EE_((y_0, y_t) tilde.op sigma_t^+) ||y_0 - (f^t_phi compose g_theta)(y_t)||^2]
$
To practically sample from this loss, we write the expectation using the given data distribution $rho^+$:
$
  cal(L)_"denoise"^+
  &:= EE_(t) [alpha_t dot EE_(x tilde.op rho^+ \ epsilon tilde.op cal(N)(0, I)) ||y_0 - (f^t_phi compose g_theta)(y_t)||^2]\
  y_t &:= overline(t) y_0 + t epsilon, quad y_0 := "stopgrad"(f_phi (x))
$
We've abbreviated $overline(t) := 1-t$. Note the stopgrad application of $f_phi$ as well: we're just using $f_phi$ to sample from the pushforward.


==== Estimated score mismatch

We use overline to denote application without gradient capture, the result still has attached gradients, but the applied function's parameters are treated as constants in the gradient graph. The score matching loss is $EE_(y_t tilde.op sigma_t^+) ||hat(y_-^*) - hat(y_+^*)||^2$ with analytic expression
$
  cal(L)_("score")^* (theta) = EE_t [alpha_t dot EE_(y_t tilde.op sigma^+_t) ||overline((f_phi^t compose g_theta))(y_t) - EE_-^*[y_0 | y_t]||^2]
$
We want to minimize this loss w.r.t. $theta$, while holding the round-trip score estimators $f_phi^(plus.minus t) compose g_theta$ fixed.

One big problem with the loss above is that #blue[$y_t tilde.op sigma_t^+$ has no generator dependency]. This problem generally plagues one-shot generation models: the forward KL (or score matching) requires optimizing $theta$ under the true data measure, but data don't have $theta$-dependency.

We introduce the *roundtrip trick*: given model latent $y=f_phi (x) tilde.op sigma^+$, it naively doesn't have $theta$-dependence. However, we can use $f_phi compose g_theta = "Id"$ on the latent support to infer which latent could have produced the data sample $x$, then the round-trip latent
$
  tilde(y) = (f_phi compose g_theta)(y)
$
has proper generator $theta$ dependency. To ensure round-trip fidelity, we need
$
  cal(L)_"roundtrip" (theta, phi) = EE_(y tilde.op sigma^+) ||(f_phi compose g_theta)(y) - y||^2
$
Using this trick, approximate
$
  tilde(cal(L))_("score") (theta) = EE_t [alpha_t dot EE_(y_t tilde.op sigma^+_t) ||overline((f_phi^t compose g_theta))(tilde(y_t)) - EE_-^*[y_0 | tilde(y)_t]||^2]
$
We optimize this objective using latent drifting (stopgrad distillation): use the gradient of estimated score-difference w.r.t. latents to guide the generated latents.
$
  V (y, epsilon, t) := -alpha_t dot nabla_y||overline((f_phi^t compose g_theta))(y_t) - EE_-^*[y_0 | tilde(y)_t]||^2, quad y_t := (1-t) y + t epsilon
$
This is a curl-free gradient field. It points latent particles $y$ in the direction which minimizes the estimated score mismatch. This is the practical decoder-side objective, and the catalog above lists this final theta-loss rather than the idealized star-loss:
$
  cal(L)_"score" (theta) & = EE_(y tilde.op sigma^+, t, \ epsilon tilde.op cal(N)(0, I)) ||"stopgrad"[
                             V_t (tilde(y), epsilon, t) + tilde(y)
                           ] - tilde(y)||^2 \
                tilde(y) & := (f_phi compose g_theta)(y)
$

==== Model-latent denoising

The model-latent denoising objective is
$
  cal(L)_"denoise"^- := EE_(t) [alpha_t dot EE_(y_t tilde.op sigma_t^+) ||EE_-[y_0 | y_t ] - EE_-^*[y_0 | y_t]||^2]
$
The analytic Gaussian denoiser is known to be
$
  EE_-^*[y_0 | y_t] = (overline(t)) / (overline(t)^2 + t^2) y_t
$
One big problem with the loss above is that #blue[$y_t tilde.op sigma_t^+$ has no generator dependency]. This problem generally plagues one-shot generation models: the forward KL (or score matching) requires optimizing $theta$ under the true data measure, but data don't have $theta$-dependency.

=== DPI bound $=>$ data reconstruction loss

The DPI term $EE_(y tilde.op sigma^+) D(P_(rho^+ | y) || P_(rho^-|y))$ can be upperbounded by the negative log-likelihood of a Gaussian decoder channel model, which reduces to a squared reconstruction loss:
$
  cal(L)_"reconstruct" (theta, phi) = EE_(x tilde.op rho^+) ||g_theta (f_phi (x)) - x||^2
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
  = 1/(2 beta) cal(L)_"reconstruct" + C
$

=== Addendum

There are many variants which might be considered; for example, $f_phi compose g_theta = "Id"$ constrains $sigma_t^- approx nu$, meaning that we can possibly get rid of all the $cal(L)^-$ machinery and substitute with closed-form Gaussian quantities; this is similar to WAE / AAE with score-matching Gaussian regularization, with the critical difference that distribution-loss changes the _encoder_ in AE-frameworks, while we use distribution mismatch loss to drift the decoder. Overall, I think there're some interesting ideas in the encoder-decoder roundtrip + Tweedie score-matching construction which might lead to, or inspire, interesting research.
