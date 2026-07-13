## Formal Model: Belief Filtering Under Bounded Policy Proxies {#sec:formal-model}

### The POMDP Surface Separates Hidden Cells from Observations {#sec:formal-discrete-model}

The active-inference analysis uses a discrete POMDP-style generative model, following the convention that hidden states generate observations and policies induce transitions [@kaelbling1998pomdp; @friston2010fep; @friston2017process; @friston2022simpler; @dacosta2020discrete_active_inference; @parr2022active_inference_book; @smith2022tutorial; @heins2022pymdp]. In this scaffold the hidden state is the discretized spatial cell from [@eq:spatial-state].

For each observation modality $m$, the likelihood tensor is:

$$
A^{(m)}_{o,s} = P(o^{(m)}=o \mid s),
\qquad
\sum_o A^{(m)}_{o,s}=1.
$$ {#eq:likelihood}

The configured transition controls define:

$$
B^{(u)}_{s',s} = P(s_{i+1}=s' \mid s_i=s, u_i=u),
\qquad
\sum_{s'} B^{(u)}_{s',s}=1.
$$ {#eq:transition}

Prior preferences and initial-state priors are represented as:

$$
C^{(m)}_o = \log P_\mathrm{pref}(o^{(m)}=o),
\qquad
D_s = P(s_0=s).
$$ {#eq:preferences-prior}

The default modality set (location, time-of-day, speed, reporting-burst), the control set, and the emitted tensor shapes are pinned in [@sec:methodology-default-config]; in the default configuration the preferences are uniform, a fact whose consequences for the policy score are made explicit below. The temporal prior used at observation $i$ is propagated explicitly:

$$
D_0(s)=D_s,
\qquad
D_i(s)=\sum_{s'} B^{(u_{i-1})}_{s,s'} q_{i-1}(s')
\quad i>0.
$$ {#eq:temporal-prior}

The active-inference diagnostic chooses $u_{i-1}$ as the argmax of the previous policy posterior [@eq:policy-posterior] (deterministic selection, not sampling), while the fixed-transition baseline filter uses `{{BASELINE_CONTROL}}` for the full sequence.

Given observed categorical outcomes $o_i$, the filtering update is:

$$
q_i(s) =
\sigma\left(
\log D_i(s) + \sum_m \log A^{(m)}_{o_i^{(m)},s}
\right),
$$ {#eq:belief-update}

where $\sigma$ normalizes a vector onto the simplex. The update is evaluated only after smoothing: model construction applies positive pseudocounts and probability floors to $D_i(s)$ and to every likelihood entry that can be logged, so [@eq:belief-update] never relies on $\log 0$.

This recursion is the common Bayesian-filtering operation: the transition predicts a distribution over hidden states and the likelihood updates that distribution after the next observation. Discrete active inference expresses it within a POMDP generative model, HMM filtering supplies the corresponding finite-state recursion, and particle filtering approximates the same operation with weighted samples when exact propagation is impractical [@dacosta2020discrete_active_inference; @kaelbling1998pomdp; @rabiner1989hmm; @arulampalam2002particle_filters].

The one-step predictive loss uses the configured conditional-independence approximation over observation modalities:

$$
P(o_i \mid s) = \prod_m A^{(m)}_{o_i^{(m)},s}.
$$ {#eq:observation-factorization}

$$
\ell_i = -\log \left(
\sum_s P(o_i \mid s) D_i(s)
\right).
$$ {#eq:predictive-loss}

Posterior uncertainty is reported as Shannon entropy [@shannon1948communication; @cover2006elements_information]:

$$
H(q_i) = -\sum_s q_i(s) \log q_i(s).
$$ {#eq:posterior-entropy}

### The Partial-Observability Protocol Tests Marginalize Versus Commit {#sec:formal-partial-observability}

The default benchmark sets the observation equal to the discretized latent cell, so the emission is the identity and a point estimate is sufficient — there is no hidden state left to marginalize. This is why no active-inference advantage survives on the default surface, and it is the central reason the formal model is introduced *before* the headline result rather than after it.

**Identity-emission collapse.** If the location emission is the identity, then observing $o_i$ sets the posterior over the latent cell to the one-hot vector $\mathbf{e}_{o_i}=\mathbf{e}_{s_i}$ after normalization. The soft predictor, hard predictor, observed-Markov predictor, and oracle-fed predictor therefore all propagate the same previous-cell vector through the same transition. Any measured advantage from recursive marginalization must disappear on that surface; a nonzero advantage requires a genuinely noisy or missing emission.

To isolate the regime in which a recursive belief over a hidden state earns its keep, let $K$ denote the number of latent cells in this protocol and let $P$ denote the learned transition matrix shared by every fair predictor. We observe a latent cell sequence through an explicit symmetric emission with flip probability $\varepsilon$,

$$
P(o_i = c \mid s_i) = (1-\varepsilon)\,\mathbb{1}\{c = s_i\} + \frac{\varepsilon}{K-1}\,\mathbb{1}\{c \neq s_i\},
$$ {#eq:po-emission}

and crucially the generator draws each flipped observation uniformly over the $K-1$ cells *other than* the true cell — a $K$-ary symmetric channel in the information-theoretic sense [@cover2006elements_information] — so the data-generating channel is exactly [@eq:po-emission] and the filter is correctly specified rather than mismatched. A standard Bayes filter maintains the belief $b_i \propto P(o_i \mid \cdot)\odot(P\,b_{i-1})$ from past observations only, and three predictors form the next-cell distribution from the *same* learned transition $P$, the *same* emission, the *same* observations, and the *same* priors, differing only in how they use the belief:

$$
\hat p^{\,\mathrm{soft}}_i = P\,b_{i-1},
\qquad
\hat p^{\,\mathrm{hard}}_i = P\,\mathbf{e}_{\arg\max b_{i-1}},
\qquad
\hat p^{\,\mathrm{obs}}_i = P\,\mathbf{e}_{o_{i-1}},
$$ {#eq:po-predictors}

where $\mathbf{e}_k$ is the $k$-th unit vector. Soft marginalizes the full belief; hard collapses it to its maximum-a-posteriori cell before propagating; observed-Markov commits to the raw noisy observation ([@eq:po-predictors]). Because soft and hard are identical except for marginalize-versus-commit, no hyperparameter can be asymmetric between them — the comparison is fair by construction, which is precisely the property the retracted observation-noise frontier of [@sec:roadmap-aif-deepening] lacked (an earlier version used an asymmetric noise knob that flattened one predictor's prior while leaving the other's untouched). Each predictor is scored by the prequential next-latent-cell negative log-loss on held-out travellers [@dawid1984prequential; @gneiting2007scoring]; the transition is estimated from training travellers only, and predictions at step $i$ use observations through $i-1$, so the comparison is leakage-free against the true latent target. The direction of this comparison is anchored by standard theory: the log score is strictly proper [@gneiting2007scoring], so among predictors evaluated under the model, the model's own posterior predictive is the unique minimizer of expected loss, and any competitor $q$ pays a model-expected excess of exactly $D_{\mathrm{KL}}(\hat p^{\,\mathrm{soft}}_i \,\|\, q) \ge 0$ — committing to the MAP cell is a lossy compression of the belief. At $\varepsilon = 0$ the belief is a one-hot vector, so marginalize and commit coincide identically; the zero-at-clean flag ({{PARTIAL_OBS_ZERO_WHEN_CLEAN}}) records an algebraic identity, not a fortunate sample. What theory does not guarantee — because the shared transition is estimated from training travellers rather than known — is the sign under the true generator at finite sample; that residual is precisely what the all-positive resampling check and the {{PARTIAL_OBS_CROSS_DRAW_COUNT}}-draw sign-stability check establish empirically. Concretely, at each traveller's first report the belief is initialized by normalizing the emission likelihood under a uniform prior over the $K$ cells, and that step is never scored; the shared transition is a column-stochastic matrix estimated from training-traveller transitions only, with a symmetric pseudocount of {{PARTIAL_OBS_SMOOTHING}} on every entry; and the MAP collapse of the hard predictor breaks exact ties toward the lowest cell index.

Uncertainty is reported as a robustness band rather than a generalization interval. Each noise level aggregates $\lvert\text{noise seeds}\rvert\times\lvert\text{split seeds}\rvert$ paired soft-versus-hard resamplings on the same latent draw; we report the mean, the full min–max envelope, the fraction of resamplings with a positive advantage (a paired sign-test proportion), and an ordinary-least-squares slope of the mean advantage against $\varepsilon$ with its $R^2$. Because this run scores a single latent draw, the spread characterizes sensitivity to the emission noise draw and the holdout split, not between-dataset variance. The summary scalars — $\varepsilon$-resolved advantage, clean-noise advantage, the noise slope and $R^2$, the worst-case and standard-deviation band at the strongest noise, and the monotonicity and all-positive flags — are bound to manuscript variables through the provenance contract of [@sec:formal-robustness].

### The Policy Score Is a Proxy, Not Full Expected Free Energy {#sec:formal-policy-proxy}

The policy-proxy diagnostic evaluates each control by propagating the current posterior through [@eq:transition] and projecting expected observations through [@eq:likelihood]. For a candidate control $u$, write $p_u^{(m)}=A^{(m)}B^{(u)}q_i$ for the predicted observation distribution and $\pi^{(m)}=\sigma(C^{(m)})$ for the configured preference distribution. The implemented score is the projected preference cross-entropy proxy:

$$
S(u)
= \sum_m
\left[
H\!\left(p_u^{(m)}\right)
+ D_{\mathrm{KL}}\!\left(p_u^{(m)} \,\middle\|\, \pi^{(m)}\right)
\right]
= \sum_m H\!\left(p_u^{(m)}, \pi^{(m)}\right)
$$ {#eq:policy-proxy}

The entropy-plus-divergence display is algebraic bookkeeping: $H(p)+D_{\mathrm{KL}}(p\|\pi)=-\sum_o p(o)\log \pi(o)$. It is not two independently estimated active-inference terms, and it is not canonical expected free energy. For one modality, a standard one-step expected-free-energy decomposition can be written as a risk-plus-ambiguity score [@millidge2020efe; @parr2022active_inference_book]:

$$
G(u)
=D_{\mathrm{KL}}\!\left(p_u \,\middle\|\, \pi\right)
+\mathbb{E}_{q(s'\mid u)}\!\left[H(A(\cdot\mid s'))\right].
$$ {#eq:efe-risk-ambiguity}

The corresponding marginal-entropy identity is:

$$
H(p_u)
=\mathbb{E}_{q(s'\mid u)}\!\left[H(A(\cdot\mid s'))\right]
+I(o;s'\mid u),
\qquad
S(u)=G(u)+I(o;s'\mid u).
$$ {#eq:proxy-efe-gap}

The implemented multimodal score in [@eq:policy-proxy] sums this modality-wise bookkeeping across $m$. The proxy therefore differs from expected free energy by exactly the sign-flipped epistemic-value term in this formal diagnostic: minimizing $S$ penalizes controls whose predicted observations are informative about the hidden state, whereas canonical expected free energy rewards them ([@eq:efe-risk-ambiguity]; [@eq:proxy-efe-gap]; [@sec:pymdp-performance-interpretability]). This is one concrete reason the pymdp lane and the proxy lane can legitimately select different controls from identical tensors ([@sec:pymdp-performance-interpretability]). We use the term proxy deliberately: this score is an implementation diagnostic over the configured tensors, not a complete derivation of expected free energy with epistemic-value terms or a normative optimal-design rule [@millidge2020efe; @sajid2021active_inference_design; @champion2024reframing_efe; @sweeney2026formal_equivalences; @vanoostrum2024concise].

Policy posterior probabilities are then:

$$
Q(u \mid o_{\leq i}) = \sigma(-S(u)).
$$ {#eq:policy-posterior}

The softmax is taken implicitly under a flat policy prior and unit precision (no temperature parameter), over a depth-one control space: controls here are single-step transition choices, not the multi-step policy sequences of the standard treatment [@friston2017process; @parr2022active_inference_book]. One degenerate case deserves explicit statement because the default run sits in it: with uniform preferences (the default and only main-lane configuration), the preference divergence reduces to the log-cardinality of each modality minus the predictive entropy, so the two displayed components cancel and the proxy score is exactly constant across controls on any data. The policy posterior is then exactly uniform (mean policy entropy {{MEAN_POLICY_ENTROPY}} nats over the two controls), the reported policy tie is an algebraic identity of uniform preferences rather than an empirical property of the fixture, and the per-step control choice of the diagnostic reduces to floating-point tie-breaking.

This completes the generative-model and partial-observability scaffold. How these tensors behave empirically across regimes is reported in the Results ([@sec:results]), and the implementation-level performance and interpretability boundary of the engine is detailed in the supplement ([@sec:pymdp-performance-interpretability]).
