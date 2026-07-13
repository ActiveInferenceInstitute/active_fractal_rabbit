# Supplementary S4 - Formal Diagnostics Stay Proxies and Provenance Checks {#sec:formal-robustness}

This supplement collects the implementation-level formalisms — robustness, privacy-proxy, and provenance contracts, plus the pymdp performance/interpretability boundary — that underpin but are not part of the main-line generative-model argument ([@sec:formal-model]).

The robustness extensions keep the same artifact discipline as the generative model of [@sec:formal-model]. A return-conditioned prior introduces a visited-destination feature

$$
r_i = \mathbb{1}\{s_i \in V_{i-1}\},
\qquad
V_i = V_{i-1} \cup \{s_i\},
$$ {#eq:return-feature}

and promotes it only when the move-step conditional entropy reduction

$$
\Delta H_\mathrm{return}
= H(s_i \mid s_{i-1}) - H(s_i \mid s_{i-1}, r_i)
$$ {#eq:return-entropy}

clears the predeclared artifact gate. Otherwise the result remains an explicit null rather than a hidden negative result. Cross-realization robustness uses empirical bands over repeated synthetic-generator realizations:

$$
\mathcal{B}_\alpha(z)
=
\left[
Q_\alpha(\{z_j\}),
Q_{1-\alpha}(\{z_j\})
\right],
$$ {#eq:variance-band}

where $z_j$ is a scalar verdict metric from realization $j$. These bands are descriptive synthetic-realization intervals, not population confidence intervals.
They should be read like bootstrap-style uncertainty summaries over generated software runs, not like a guarantee that the same interval would cover a real-world population quantity [@efron1994bootstrap].

The privacy frontier reports only oversight and minimization quantities. For traveller $a$, a coarsened signature is

$$
\Gamma_a(M)=\{(\tau_i, c_i^M): \mathrm{traveller}(i)=a\},
$$ {#eq:signature}

where $M$ is a modality-minimization regime and $c_i^M$ is the retained categorical code. Synthetic unicity and co-travel coincidence are then

$$
U(M)=\frac{|\{\Gamma_a(M)\}|}{|\{a\}|},
\qquad
C(M)=\frac{1}{|\mathcal{P}|}\sum_{(a,b)\in\mathcal{P}}
\mathbb{1}\{\Gamma_a(M)\cap\Gamma_b(M)\neq\varnothing\}.
$$ {#eq:privacy-frontier}

The set-valued signature is intentional: it discards multiplicity and ordering to make a coarse oversight proxy, not a full reconstruction attack, k-anonymity mechanism, differential-privacy mechanism, or formal privacy proof [@sweeney2002kanonymity; @dwork2006differential_privacy; @buchholz2024trajectory_privacy_sok]. If a later analysis prices repeated visits, ordered routines, or temporal sequence alignment, $\Gamma_a(M)$ must become a multiset or sequence and the corresponding frontier must be regenerated.

The hardened anomaly probe compares pymdp-style surprise with a fair online Markov-surprise baseline:

$$
\mathcal{S}_i^\mathrm{Markov}
= -\log \widehat{P}_{i-1}(s_i \mid s_{i-1}),
$$ {#eq:markov-surprise}

and reports both AUC and operating-point metrics so separability is not summarized by area alone. Finally, result-critical manuscript tokens are bound to source artifacts by a provenance tuple

$$
\pi_k = (k, p_k, \eta_k, h_k, v_k),
$$ {#eq:token-provenance}

where $k$ is the token, $p_k$ the source artifact path, $\eta_k$ the JSON key path, $h_k$ the source checksum, and $v_k$ the rendered value. Rendering is trusted only if the artifact exists, the key resolves, the checksum is current, and the formatted artifact value byte-matches the token.

## pymdp Adds Interpretability Without Predictive Leadership {#sec:pymdp-performance-interpretability}

The same `A`, `B`, `C`, and `D` arrays are also consumed by the actual pinned `pymdp` agent. The benchmark matrix reports `{{PYMDP_SOTA_STATUS}}` for state-of-the-art performance and `{{PYMDP_INTERPRETABILITY_STATUS}}` for diagnostic value — a distinction that is load-bearing. `pymdp` currently helps by exposing variational free energy, policy posteriors, and expected-free-energy terms in a common active-inference grammar; reaching performance leadership would require clearing the sequence-baseline gate in [@sec:results-sequence-reporting-gates] and the broader roadmap gates in [@sec:roadmap-sota-gates], not just replacing the transparent reference filter with the library agent.

When `model.engine` is `both`, the pipeline constructs a batched JAX `Agent` from the tensors above and runs a sequential perception-action loop: variational state inference under the current empirical prior, policy inference returning the policy posterior and expected free energy, deterministic action selection, and an empirical-prior update through the selected control's transition slice [@heins2022pymdp; @pymdp2026docs; @pymdp2026jaxmigration]. Two comparisons are emitted. First, a matched-prior validation: at every step the pymdp variational posterior is compared against the exact Bayes posterior computed from the same prior and tensors; on the default run the maximum absolute difference is {{PYMDP_EXACT_INFERENCE_DELTA}}, which validates the inference implementations against each other. Second, a closed-loop comparison: the full trajectories diverge (maximum absolute posterior difference {{PYMDP_LOOP_POSTERIOR_DELTA}} on this run) through policy-induced prior divergence — differing action-selection criteria (expected free energy versus the preference-cross-entropy proxy score of [@eq:policy-proxy]) select different controls, which propagate different empirical priors, so subsequent posteriors legitimately differ. Because the posteriors agree under matched priors at every step, the closed-loop divergence is fully explained by differing action sequences rather than any latent inference discrepancy. Its magnitude is trajectory-specific, not a stable property of the implementations. Both traces are emitted side by side in `output/runs/default/pymdp_trace.csv` and `output/runs/default/policy_trace.csv`. The pymdp trace additionally records, per timestep, the agent's own variational free energy (the final fixed-point iterate returned by `infer_states`), the wall-clock step runtime, the policy posterior, and the expected free energy of every policy, so the full diagnostic surface of the pinned library is a run artifact rather than an internal quantity.
