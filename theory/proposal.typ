#let hyperlink-blue = rgb("#1f4b99")
#let blue(body) = text(fill: hyperlink-blue, body)
#show link: it => underline(
  stroke: hyperlink-blue,
  text(fill: hyperlink-blue, it.body),
)

= Chart transport proposal
Nicholas (Xingjian) Lyu. Mar 23, 2026
\

*TLDR*: we propose a generative modeling protocol which #blue[_sidesteps density estimation variance in high-dimensions_] and #blue[_performs score matching_] under the correct data measure #footnote[Score matching under the model distribution minimizes the mode-seeking #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#ex-otto-reverse-kl")[reverse KL], instead of the proper MLE objective.]. The protocol admits a clean MLE interpretation.

The main mathematical tools are #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#thm-tweedie")[Tweedie's formula] and #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#prp-fm")[de Bruijn's identity]. In my opinion, they are the mathematical powerhouse behind the stability and dimension-scalability of flow matching models. We couple these tools with one-step generation by using an encoder-decoder construction #footnote[equivalently, generator-critic; there are many perspectives here]. Main highlights of the method:

1. Low-variance forward-KL optimization without importance sampling.
2. Robustness under weak data-model support overlap in high dimensions.
3. Scalable high-dimensional density estimation via score-matching.

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
  "latent" stretch(harpoons.rtlb)^(g_theta)_(f_phi) "model sample"
$
Define the ambient manifold of all encoder-decoder pairs
$
  cal(A) = { (f_phi, g_theta) }
$
We use #blue[$rho^+$ for the data distribution, $rho^- = g_(theta\#) (nu)$ for the model distribution, and $x tilde.op rho^plus.minus$ for samples]. We also write #blue[$sigma^plus.minus = f_(phi\#) (rho^plus.minus)$ for the latent distributions, and $y tilde.op sigma^plus.minus$ for latents.]
$
  "latent" nu arrow.r^(g_theta) "sample" rho^- arrow.r^(f_phi) sigma^-, quad "true data" rho^+ arrow.r^(f_phi) sigma^+
$

Define the *chart manifold* $cal(M)subset cal(A)$ by two support identities
$
  cal(M) = {
    (f_phi, g_theta) in cal(A):
    (g_theta compose f_phi)(x) = x " on " rho^+,
    quad
    (f_phi compose g_theta)(z) = z " on " nu
  }
$
On $cal(M)$, the model latent is analytic: the prior-side identity gives $sigma^- = nu = cal(N)(0, I)$. The sample-space MLE objective therefore reduces to pushing the data latent $sigma^+$ towards the Gaussian prior while staying on the chart manifold. Define the noised latent processes by the canonical linear interpolation with $t in [0, 1]$:
$
  sigma^plus.minus_t = (1-t) sigma^plus.minus + t cal(N)(0, I)
$
Since $sigma^- = nu$, the model-side marginals remain closed-form:
$
  sigma^-_t = cal(N)(0, (t^2 + overline(t)^2) I)
$

We use #blue[a separate noise critic $s_psi$ to learn the score of the noised data latent $sigma^+_t$]. The model-side score is analytic, so only the data-side score must be learned. The latent score difference
$
  delta_t (y_t) := nabla log sigma^+_t (y_t) - nabla log sigma^-_t (y_t)
$
is the Wasserstein gradient of latent KL respect to the flowing data latent. We estimate this field with the critic, pull it back to the clean latent, and update the chart pair in a projection-like way so that both encoder and decoder receive first-principles supervision.

=== High-level objectives <high-level-objectives>
The detailed losses which we'll proceed to developing all serve the following objectives:

1. Manifold constraint: $(g_theta compose f_phi)(x) = x$ on $rho^+$ and $(f_phi compose g_theta)(z) = z$ on $nu$.
2. Critic correctly estimates the noised data-latent score: $s_psi (y_t, t) approx nabla log sigma^+_t (y_t)$.
3. Transport the chart along the sampled score-difference field while remaining on $cal(M)$:
$
  & EE_(x tilde.op rho^+, t, epsilon tilde.op cal(N)(0, I))
    ||f_phi(x) - "sg"(y - eta dot overline(t)alpha_t dot delta_t (y_t))||^2 \
  & EE_(x tilde.op rho^+, t, epsilon tilde.op cal(N)(0, I))
    ||g_theta ("sg"(y - eta dot overline(t)alpha_t dot delta_t (y_t))) - x||^2
$
where $y := "sg"(f_phi(x))$ and $y_t := overline(t) y + t epsilon$, and $alpha_t$ is bulk-KL weighting.


== Theory

=== Chart manifold reduces sample-KL to latent-KL

Fix $(f_phi, g_theta) in cal(M)$. The data-side identity implies that $f_phi$ is injective on the data support: if $f_phi (x_1) = f_phi (x_2)$ with $x_1, x_2 in "supp"(rho^+)$ then
$
  x_1 = (g_theta compose f_phi)(x_1) = (g_theta compose f_phi)(x_2) = x_2
$
The prior-side identity similarly makes $f_phi$ injective on the model support, and also gives
$
  sigma^- = f_(phi\#) rho^- = (f_phi compose g_theta)_(\#) nu = nu
$
Therefore pushing both sample distributions through the injective encoder preserves KL:
$
  D(rho^+ || rho^-)
  = D(f_(phi\#) rho^+ || f_(phi\#) rho^-)
  = D(sigma^+ || sigma^-)
  = D(sigma^+ || nu)
$
So #blue[on the chart manifold, sample-space MLE is exactly latent-space Gaussianization]. The DPI residual disappears because the encoder is information-lossless on the relevant supports.

=== Noise spectrum and critic

Define the noised data latent process by
$
  y_t = overline(t) y_0 + t epsilon,
  quad y_0 tilde.op sigma^+,
  quad epsilon tilde.op cal(N)(0, I)
$
and similarly for the model latent. Because $sigma^- = nu$, the noised model latent is analytic:
$
  sigma^-_t = cal(N)(0, (overline(t)^2 + t^2) I),
  quad
  nabla log sigma^-_t (y_t) = - y_t / (overline(t)^2 + t^2)
$

Now recall from #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#weighted-bulk-kl")[part 3] that a free-form score-difference weighting induces a corresponding weighted bulk-KL. Write
$
  cal(B)_alpha (sigma^+)
  := EE_(t tilde.op "Unif"[0, 1]) [
    alpha_t EE_(y_t tilde.op sigma_t^+) ||nabla log sigma^+_t (y_t) - nabla log sigma^-_t (y_t)||^2
  ]
$
The exact MLE choice $alpha_t = t slash overline(t)$ collapses this spectrum back to the endpoint KL $D(sigma^+ || nu)$. Ordinary uniform-in-time velocity matching corresponds to a bulk-KL weighting proportional to $1 / overline(t)^2$. We'll keep the schedule $alpha_t$ free, because the chart-transport update can inherit any desired noise-spectrum weighting.

Only the data-side score must be estimated. We use a separate critic $s_psi$ trained on noised data latents:
$
  cal(L)_"critic" (psi; phi)
  := EE_(t tilde.op "Unif"[0, 1], x tilde.op rho^+, epsilon tilde.op cal(N)(0, I))
  ||s_psi (y_t, t) - epsilon||^2,
  quad
  y_t := overline(t) y + t epsilon,
  quad
  y := "sg"(f_phi (x))
$
Its implied score estimate is
$
  s_psi (y_t, t) := - (s_psi (y_t, t)) / t
$
so the sampled score-difference field is
$
  delta_t (y_t) := s_psi (y_t, t) - nabla log sigma^-_t (y_t)
$

=== Projected Wasserstein descent on charts

At each fixed $t$, the Wasserstein gradient of $D(sigma_t^+ || sigma_t^-)$ with respect to the flowing first argument $sigma_t^+$ is precisely the score difference $delta_t (y_t)$. So latent descent follows $-delta_t (y_t)$.

However, our trainable object is not the free latent distribution; it is the chart pair $(f_phi, g_theta)$ constrained to lie on $cal(M)$. Since $y_t = overline(t) y + t epsilon$, pulling the noised-latent descent back to the clean latent gives
$
  nabla_y = overline(t) nabla_(y_t)
  quad => quad
  "clean-latent descent" = - overline(t) delta_t (y_t)
$
Thus a sampled latent target is
$
  tilde(y)
  := y - eta alpha_t overline(t) dot delta_t (y_t),
  quad
  y := "sg"(f_phi (x)),
  quad
  y_t := overline(t) y + t epsilon
$
This ambient Wasserstein step will generally leave the chart manifold, so we update the chart pair by fitting to the drifted latent target while also enforcing the two manifold constraints. This is our projection-like descent on $cal(M)$:
$
         cal(L)_"cycle"^+ (theta, phi) & := EE_(x tilde.op rho^+) ||(g_theta compose f_phi)(x) - x||^2 \
         cal(L)_"cycle"^- (theta, phi) & := EE_(z tilde.op nu) ||(f_phi compose g_theta)(z) - z||^2 \
  cal(L)_"chart-enc" (phi; theta, psi) & := EE_(x, t, epsilon) ||f_phi(x) - "sg"(tilde(y))||^2 \
  cal(L)_"chart-dec" (theta; phi, psi) & := EE_(x, t, epsilon) ||g_theta ("sg"(tilde(y))) - x||^2
$
The encoder loss moves the data latent in the projected Wasserstein descent direction. The decoder loss makes the same drifted latent continue to decode to the original data sample. The cycle losses keep the pair on the chart manifold so that the latent objective continues to equal the sample-space MLE objective.
