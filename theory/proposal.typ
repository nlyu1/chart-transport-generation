#let hyperlink-blue = rgb("#1f4b99")
#let blue(body) = text(fill: hyperlink-blue, body)
#show link: it => underline(
  stroke: hyperlink-blue,
  text(fill: hyperlink-blue, it.body),
)
#set math.equation(numbering: "(1)")

= Chart transport proposal
Nicholas (Xingjian) Lyu. Mar 23, 2026
\

*TLDR*: we propose a generative modeling protocol which #blue[_sidesteps density estimation variance in high-dimensions_] and #blue[_performs score matching_] under the correct data measure #footnote[Score matching under the model distribution minimizes the mode-seeking #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#ex-otto-reverse-kl")[reverse KL], instead of the proper MLE objective.]. The protocol admits a clean MLE interpretation.

The main mathematical tools are #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#thm-tweedie")[Tweedie's formula] and #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#prp-fm")[de Bruijn's identity]. In my opinion, they are the mathematical workhorse behind the stability and dimension-scalability of flow matching models. We apply these tools to one-step generation using an encoder-decoder construction. Highlights:

1. Low-variance forward-KL optimization without importance sampling.
2. Robustness under weak data-model support overlap in high dimensions.
3. Scalable high-dimensional density estimation via score-matching.

== Interpretation of drifting; challenges

We have shown in #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#gaussian-kernel-smoothing-implements-reverse-kl")[these notes] that drifting implements Wasserstein gradient descent on the reverse-KL divergence $D(rho_"model" || rho_"data")$ using kernel density estimates (KDE) of the model and data densities. The drifting field is the Gaussian kernel-estimate of score difference between data and model distributions at that point#footnote[#link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#ex-otto-reverse-kl")[Applying Otto's theorem to reverse KL] shows that the score-difference field on sample space is the Wasserstein gradient of reverse KL.]. This perspective suggests the following challenges with existing drifting methods:

1. *Gaussian KDE variance scales with data dimension*: this is one possible explanation for why ImageNet drifting does not work without a good pretrained encoder.
2. *KL objective is stiff*: KL blows up when supports don't overlap. When data and model distributions don't overlap, drifting fails because the fundamental Wasserstein-functional objective blows up, _even if density estimates were perfect_. This is another possible explanation for why drifting struggles to natively scale to high dimensions.
3. *The reverse-KL sampling problem* #footnote[See also #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#proposition-maximum-likelihood-drifting")[MLE drifting].]: it's well-known in generative modeling and RL literature  that optimizing the reverse KL $D(rho_"model" || rho_"data")$ is prone to mode collapse and low diversity. However, optimizing the proper forward-KL objective is hard because it requires sampling the model w.r.t. the _data distribution_ $EE_"data"$, while reverse-KL samples the model naturally under $EE_"model"$. Typical one-shot models cannot sample from $EE_"data"$ _because we can't tell which latent could have generated a data sample_.


== Escape hatches

I did #link("http://localhost:4321/blogs/machine-learning/ot-generative-3-diffusion/")[some learning on diffusion / flow-matching] to see how these problems are addressed in the FM paradigm. The main takeaways are:

1. _Reduce density (score) estimation to regression_: #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#thm-tweedie")[Tweedie's formula] reduces score estimation to denoising #footnote[$x \| z tilde.op cal(N)(alpha z, Sigma)$]. The difficulty of parametric regression scales with data size and the structural (manifold) dimension of the problem instead of the data dimension.
$
  alpha EE[z \| x] = x + sigma^2 nabla log rho
$
2. _Decompose KL along the noise spectrum into score matching_: by decomposing KL into a score matching integral, we can make powerful bias-variance tradeoffs to e.g. truncate the divergent $t->0$ component, reweight score matching losses, and importance sample, etc#footnote[The de Bruijn integral we presented assumes the (dominantly used) flow matching process, which corresponds to optimal-transport of the independent data-noise coupling.]

== High-level proposal

Our proposal is motivated by applying the escape hatches above to drifting with MLE in mind. From first principles, properly #footnote[we could consider importance sampling, but it introduces high variance and dependence upon explicit density estimation of $rho_"model" slash rho_"data"$.] solving the reverse-KL sampling problem requires us to know _what latent would have generated a data sample_; this calls for an inverse $f_phi$ to the generator $g_theta$. We  propose an encoder-decoder architecture
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
On $cal(M)$, the model latent is analytic: the prior-side identity gives $sigma^- = nu = cal(N)(0, I)$. Sample-space MLE objective reduces to pushing the data latent $sigma^+$ towards the Gaussian prior while staying on the chart manifold (this is made precise by the data-processing inequality). Define the noised latent processes by the canonical linear interpolation with $t in [0, 1]$:
$
  sigma^plus.minus_t = (1-t) sigma^plus.minus + t cal(N)(0, I)
$
Since $sigma^- = nu$, the model-side marginals remain closed-form:
$
  sigma^-_t = cal(N)(0, (t^2 + overline(t)^2) I)
$

We use #blue[a separate noise critic $epsilon_psi$ to learn the score $nabla log sigma^+_t (y_t)$ of the noised data latent]. The model-side score is analytic, so only the data-side score must be learned. The latent score difference
$
  nabla log sigma^+_t (y_t) - nabla log sigma^-_t (y_t)
$
is the per-time forward-KL $D(sigma_t^+ || sigma_t^-)$ Wasserstein gradient with respect to the flowing data latent. We can use this to estimate the clean-latent transport field $delta(y)$ and update the chart pair.

=== High-level objectives <high-level-objectives>
The detailed losses which we'll proceed to developing all serve the following objectives:

1. Manifold constraint: $(g_theta compose f_phi)(x) = x$ on $rho^+$ and $(f_phi compose g_theta)(z) = z$ on $nu$.
2. Critic correctly estimates the noised data-latent score: $hat(s)_psi (y_t, t) approx nabla log sigma^+_t (y_t)$.
3. Transport the chart along the clean-latent field $delta (y)$ while remaining on $cal(M)$. In practice we work with an estimator $hat(delta) (y) approx delta (y)$ and fit the chart to the induced latent step:
$
  & EE_(x tilde.op rho^+)
    ||f_phi (x) - "sg"(y + eta dot hat(delta) (y))||^2 \
  & EE_(x tilde.op rho^+)
    ||g_theta ("sg"(y + eta dot hat(delta) (y))) - x||^2
$
where $y := "sg"(f_phi (x))$ and $alpha_t$ is bulk-KL weighting.


== Theory

=== DPI decomposition and chart manifold

Before imposing chart constraints, the sample-space forward KL decomposes along the encoder channel by the #link("https://nlyu1.github.io/classical-info-theory/kullback-leibler-divergence.html#chain-rule-dpi")[chain rule / DPI]:
$
  D(rho^+ || rho^-)
  = D(sigma^+ || sigma^-)
  + EE_(y tilde.op sigma^+) D(P_(rho^+ | y) || P_(rho^- | y))
$
The first term is the latent-space forward KL, and the residual term measures how much information the encoder discards about the sample-space distinction between data and model. Fix $(f_phi, g_theta) in cal(M)$. The data-side identity implies that $f_phi$ is injective on the data support, upon which the residual term vanishes
$
  D(rho^+ || rho^-)
  = D(f_(phi\#) rho^+ || f_(phi\#) rho^-)
  = D(sigma^+ || sigma^-)
  = D(sigma^+ || nu)
$
So #blue[on the chart manifold, sample-space MLE is exactly latent-space Gaussianization]. Equivalently, the DPI residual vanishes because the encoder is information-lossless on the relevant supports.

=== Bulk-KL weighting

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

The MLE target is the endpoint forward KL $D(sigma^+ || sigma^-) = D(sigma^+ || nu)$. More generally, we choose a bulk-KL weighting and define
$
  cal(K)_alpha (sigma^+)
  := integral
  alpha_t D(sigma_t^+ || sigma_t^-)
  dif t
$
This is the population-objective whose Wasserstein gradient defines the latent field we want to descend. Otto calculus identifies the corresponding Wasserstein gradient
$
  ["grad"_(W, sigma^+) cal(K)_alpha (sigma^+)](y) & = integral alpha_t overline(t) dot EE_(y_t tilde sigma_t^+(dot | y)) ["grad"_(W, sigma_t^+) D(sigma_t^+ || sigma_t^-)]_(y_t) \
  & = integral alpha_t overline(t) dot EE_(y_t tilde sigma_t^+(dot | y)) [nabla log sigma_t^+ - nabla log sigma_t^-]_(y_t) \
  &:= -delta(y)
$
This is the clean-latent Wasserstein gradient direction of the latent bulk-KL objective.

Note that the flowing measure matters. In most one-step generative modeling setups, the flowing measure is the model distribution and the score-difference field is sampled under the model measure, giving reverse-KL behavior #footnote[#link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#gaussian-kernel-smoothing-implements-reverse-kl")[Reverse-KL drifting] and #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-2-drifting-models/#implementing-forward-kl")[forward-KL via density-ratio reweighting].]. Here the flowing measure is the encoded data latent, so the same score difference appears as the forward-KL Wasserstein gradient of the first argument and must be sampled under $y_t tilde.op sigma_t^+$. Sampling instead under $sigma_t^-$ would optimize the wrong functional.

Weighted score matching is just an equivalent representation of this bulk objective. Recall from #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#weighted-bulk-kl")[part 3] that if $a_t$ weights the score-difference integral and $w_t := overline(t) slash t dot a_t$, then
$
  EE_t [a_t EE_(y_t tilde.op sigma_t^+) ||delta_t (y_t)||^2]
  = - w_t D(sigma_t^+ || sigma_t^-)|_0^1
  + EE_t [dot(w)_t D(sigma_t^+ || sigma_t^-)]
$
The exact MLE choice is the exceptional boundary case $a_t = t slash overline(t)$, which collapses back to $D(sigma^+ || nu)$. #blue[Ordinary uniform-in-time velocity matching corresponds instead to the bulk-KL weighting $alpha_t = 1 slash overline(t)^2$]. While $alpha_t$ could be kept free w.l.o.g, we canonically choose $alpha_t=(1-t)^(-2)$ corresponding to the  bulk-KL objective of uniform velocity flow-matching.

=== Score and transport field estimation
Only the data-side score must be estimated. We parameterize score via a noise predictor $epsilon_psi$ on noised data latents. We use $f_phi$ to sample $y tilde.op sigma^+$ and minimize the score critic loss:
$
  cal(L)_"critic" (psi)
  := EE_(t tilde.op "Unif"[0, 1], x tilde.op rho^+, epsilon tilde.op cal(N)(0, I))
  ||epsilon_psi (y_t, t) - epsilon||^2,
  quad
  y_t := overline(t) y + t epsilon,
  quad
  y := "sg"(f_phi (x))
$
Its #link("https://snakamoto404.github.io/blogs/machine-learning/ot-generative-3-diffusion/#flow-matching-in-practice")[implied score estimate] is
$
  hat(s)_psi (y_t, t) := - (epsilon_psi (y_t, t)) / t
$
It remains to estimate $delta(y) = -["grad"_(W, sigma^+) cal(K)_alpha (sigma^+)](y)$:
$
  delta(y)
  & = integral alpha_t overline(t) dot EE_(y_t tilde sigma_t^+(dot | y)) [nabla log sigma_t^+ - nabla log sigma_t^-]_(y_t)
$
Using our score critic, the functional approximation is
$
  hat(delta) (y) & := EE_t [1/(1-t) EE_(y_t | y) [ nabla log sigma_t^-(y_t) + (epsilon_psi (y_t, t)) / t]]
$ <eq-hat-delta>

=== Projected Wasserstein descent on charts

Our trainable object is not the free latent distribution; it is the chart pair $(f_phi, g_theta)$ constrained to lie on $cal(M)$. At the chart-transport level, define the latent target
$
  tilde(y)
  := y + eta hat(delta) (y),
  quad
  y := "sg"(f_phi (x))
$
This ambient Wasserstein step will leave the chart manifold, so we update the chart pair by fitting to the transported latent target while also enforcing the two manifold constraints.

=== Loss catalog

The full optimization protocol consists of five losses, using $hat(delta)(y)$ from @eq-hat-delta:
$
  cal(L)_"cycle"^+ (theta, phi) & := EE_(x tilde.op rho^+) ||(g_theta compose f_phi)(x) - x||^2 \
  cal(L)_"cycle"^- (theta, phi) & := EE_(z tilde.op nu) ||(f_phi compose g_theta)(z) - z||^2 \
          cal(L)_"critic" (psi) & := EE_(t, epsilon, x tilde.op rho^+)
                                  ||epsilon_psi (y_t, t) - epsilon||^2,
                                  quad
                                  y_t := overline(t) "sg"(f_phi (x)) + t epsilon \
             cal(L)_"enc" (phi) & := EE_(x tilde.op rho^+) ||f_phi (x) - tilde(y)_x||^2 \
           cal(L)_"dec" (theta) & := EE_(x tilde.op rho^+) ||g_theta (tilde(y)_x) - x||^2
$
where $tilde(y)_x = "sg"(y + eta dot hat(delta) (y))$, and $y := "sg"(f_phi (x))$.

== Implementation details

In this section, we outline potentially important implementation choices that are anticipated by the theory.

=== Two-time updates

Chart transport is contingent upon a high-quality critic. If the critic is biased, the $nabla log sigma_t^-$ term still analytically provides model-side potential, but the data-side dispersion will be biased. Combined with the fact that any changes to the chart induces non-stationarity in the critic _data distribution_ as well as _target_, we need to:

1. Start by enforcing $cal(L)_"cycle"^plus.minus$ to end up in a stable point on the manifold before training $cal(L)_"critic"$.
2. One update to the chart should be followed by several updates to the critic.

=== Lagrangian constraints (optional)

Our losses can be grouped into two categories: _objectives_ versus _constraints_. Of these, $cal(L)_"cycle"^plus.minus$ have clean interpretation as constraints, so it feels natural to parameterize them as such instead of choosing loss weightings (which is still equally valid). A principled starting place is pretraining an autoencoder with minimal prior-anchoring (e.g. mean-zero, minor latent norm penalty), recording the deviations, then use augmented Lagrangian with "budget" being some scalar multiple (e.g. 2) of the stable reconstruction losses.

=== Transport field estimate

Chart updates are expensive because they destabilize the critic. To this end, we want the field estimate to be low-variance. Recalling
$
  hat(delta) (y) & := EE_t [1/(1-t) EE_(y_t | y) [ nabla log sigma_t^-(y_t) + (epsilon_psi (y_t, t)) / t]]
$
We can consider three optimizations:

1. Use a pre-specified noise-spectrum time-grid $t_0, ..., t_m$ to approximate the integral. Also, with the factorized prior below the model-term expectation reduces to a shared one-dimensional function of $(y_i, t)$ that can be pre-tabulated.
2. Use independent $epsilon$ for each sample and timestep.
3. Evaluate for antithetic pairs $plus.minus epsilon$.

=== Prior symmetry-breaking

Naive i.i.d. Gaussian prior has continuous rotational symmetry. Any continuous symmetry of the prior aggravates non-stationarity of the chart because the encoder-decoder can freely adapt (could be driven by random sampling noise) the chart along the gauge degree of freedom. To scalably address this problem in high dimensions, we can replace the spherical Gaussian reference prior by a symmetry-broken factorized prior.

==== Factorized Gaussian scale-mixture

We replace the isotropic Gaussian prior by an i.i.d. product of a two-scale Gaussian mixture. Let $lambda >= 1$ denote the precision of the low-variance component, and write $q_v (u)$
for the one-dimensional Gaussian density with variance $v$. Define
$
  p_lambda (u)
  := lambda/(lambda + 1) q_(1 slash lambda) (u)
  + 1/(lambda + 1) q_lambda (u)
$
and the factorized latent prior
$
  nu_lambda (y)
  := product_(i=1)^d p_lambda (y_i)
$
This family has a single hyperparameter. The narrow component has variance $lambda^(-1)$. The wide component is canonically fixed to variance $lambda$, and the mixture ratio $lambda:1$ is the unique symmetric choice that preserves unit variance:
$
  EE[Y] = 0,
  quad
  EE[Y^2]
  = lambda/(lambda + 1) lambda^(-1)
  + 1/(lambda + 1) lambda
  = 1
$

==== Analytic scores under the independent-coupling process

Let the model latent follow the same independent-coupling noising process
$
  y_t = overline(t) y_0 + t epsilon,
  quad
  y_0 tilde.op nu_lambda,
  quad
  epsilon tilde.op cal(N)(0, I)
$
Because $nu_lambda$ factorizes, each coordinate remains a two-scale Gaussian mixture after noising. Define the two noised component variances
$
  v_-(t) := t^2 + overline(t)^2 lambda^(-1),
  quad
  v_+(t) := t^2 + overline(t)^2 lambda
$
Then the one-dimensional noised marginal is
$
  p_(lambda, t)(u)
  := lambda/(lambda + 1) q_(v_-(t))(u)
  + 1/(lambda + 1) q_(v_+(t))(u)
$
and the full noised prior remains factorized:
Differentiating the Gaussian kernel gives $partial_u q_v(u) = -u q_v(u) / v$, hence the one-dimensional score is
$
  s_(lambda, t)(u)
  := partial_u log p_(lambda, t)(u)
  = -u
  (lambda v_-(t)^(-1) q_(v_-(t))(u) + v_+(t)^(-1) q_(v_+(t))(u))
  /
  (lambda q_(v_-(t))(u) + q_(v_+(t))(u))
$
Equivalently, if
$
  r_-(u, t)
  := lambda q_(v_-(t))(u) / (lambda q_(v_-(t))(u) + q_(v_+(t))(u)),
  quad
  r_+(u, t) := 1 - r_-(u, t)
$
are the posterior responsibilities of the narrow and wide component, then
$
  s_(lambda, t)(u)
  = -u (r_-(u, t) / v_-(t) + r_+(u, t) / v_+(t))
$
The full model-side score is coordinatewise:
$
  nabla log nu_(lambda, t)(y_t)
  = (s_(lambda, t)(y_(t, 1)), ..., s_(lambda, t)(y_(t, d)))
$
Conditioning on the clean latent $y$, the model term entering @eq-hat-delta reduces to a shared one-dimensional function
$
  F_(lambda, t)(u)
  := EE_(epsilon tilde.op cal(N)(0, 1))
  [s_(lambda, t)(overline(t) u + t epsilon)]
$
applied coordinatewise:
$
  EE_(y_t tilde.op nu_(lambda, t)(dot | y))
  [nabla log nu_(lambda, t)(y_t)]
  = (F_(lambda, t)(y_1), ..., F_(lambda, t)(y_d))
$
The prior is non-Gaussian, but the full model-side contribution still reduces to evaluating one odd scalar function on each coordinate.

==== KL-curvature to rotation

We quantify rotational anchoring by the KL-curvature of axis-aligned rotation at $theta = 0$.

Fix one noise level $t$, and let $X_1, X_2$ be i.i.d. with marginal $p_(lambda, t)$. Rotate the pair by angle $theta$:
$
  X_1^theta := cos theta X_1 + sin theta X_2,
  quad
  X_2^theta := - sin theta X_1 + cos theta X_2
$
Let $q_theta$ denote the density of $(X_1^theta, X_2^theta)$, and define the exact pairwise KL along this rotation orbit
$
  cal(K)_(lambda, t)(theta)
  := D(q_theta || q_0)
$
Because $theta = 0$ is the aligned point, $cal(K)_(lambda, t)(0) = 0$ and the first nontrivial term is the curvature
$
  kappa_(lambda, t)
  := dif^2/(dif theta^2) cal(K)_(lambda, t)(theta)
$
Writing $s := s_(lambda, t)$, direct differentiation of $q_theta (x_1, x_2) = p_(lambda, t)(cos theta x_1 - sin theta x_2) p_(lambda, t)(sin theta x_1 + cos theta x_2)$ gives the score of the rotation family
$
  dot(ell)(x_1, x_2)
  := partial_theta log q_theta(x_1, x_2)
  = x_1 s(x_2) - x_2 s(x_1)
$
At the aligned point, the KL Hessian is the Fisher information of this one-parameter family:
$
  kappa_(lambda, t)
  = EE[dot(ell)(X_1, X_2)^2]
  = EE[(X_1 s(X_2) - X_2 s(X_1))^2]
$
Using independence of $X_1, X_2$,
$
  kappa_(lambda, t)
  = 2 EE[X^2] EE[s(X)^2] - 2 EE[X s(X)]^2
$
Now $EE[X^2] = v_t := overline(t)^2 + t^2$, while integration by parts gives
$
  EE[X s(X)]
  = integral x partial_x p_(lambda, t)(x) dif x
  = -1
$
Therefore
$
  kappa_(lambda, t)
  = 2 (v_t I(p_(lambda, t)) - 1),
  quad
  I(p_(lambda, t)) := EE[s_(lambda, t)(X)^2]
$
At $t = 0$, we have $v_0 = 1$, so the clean-prior rotational anchor is
$
  kappa_lambda
  := kappa_(lambda, 0)
  = 2 (I(p_lambda) - 1)
$
By the one-dimensional Fisher information inequality, $I(p_lambda) >= 1$ with equality iff $p_lambda$ is Gaussian. Hence $kappa_lambda > 0$ for every $lambda != 1$: the adopted prior eliminates all continuous rotational zero modes. Under noising, $p_(lambda, t)$ continuously approaches the Gaussian as $t -> 1$, so $kappa_(lambda, t)$ decays toward $0$ at high noise but remains strictly positive on every finite-noise slice whenever $lambda != 1$.
