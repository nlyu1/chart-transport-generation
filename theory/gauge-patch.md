One big problem, if we look at the multimodal chart transport artifacts, is that it
relies heavily on us getting the "intrinsic dimension" of the data correct. In particular,
we fail miserably when we try to model an inherently two-dimensional manifold inside a
high-ambient (e.g. 128-dimensional space).

To address this problem, I have the following proposal.

1. Parameterize the encoder as $f_\phi(x, \eta)$, where epsilon has the same        dimension as the latent dimension (epsilon can just be concatenated along with x). Here, epsilon denotes the "gauge" degree of freedom, and the image of $f_\phi(x)$ over epsilon defines a "fiber" in the latent space. Another perspective here is that we have a stochastic encoder. We specify that $\eta$ is drawn from i.i.d. Gaussian.
2. Next, I require the following properties:
    - Data reconstruction is replaced with the obvious generalization that the latent fibers of data are disjoint in latent space:
    $$
        g_\theta(f_\phi(x, \eta)) \approx x
    $$
    - Model latent matches the prior. In the deterministic case, we were able to enforce this by a deterministic roundtrip (that was actually too strong); here, we need the full power of score-matching. We use the same score-matching mechanism to enforce that the latent is aligned with the prior.
3. So the moving parts, and the objectives, decompose into the following:
    - Data-side reconstruction (see above).
        - This constitutes chart pretraining. Please note that we should put the weak-prior-penalty (L2) on both the data-side latent as well as the model-side latent, since both should be initialized to the prior.
    - Critic score matching. Our critic should accept a categorical variable $\pm$ corresponding to data or prior. Semantically, it should tract the score of the data and model pushforward under the stochastic $f_\phi$ kernel. Note that, critically,
    - Drifting: this is fairly straightforward, adapt the charts such that both latents drift towards the prior.

The one piece of good news is that there are almost zero more hyperparameters to set, because the data-side score-matching is the same as model-side score matching.

In terms of artifacts, we'll be looking for both data-latent and model-latents, together with transported fields.