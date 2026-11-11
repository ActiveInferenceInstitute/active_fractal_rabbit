# Documentation Index Maps Evidence to Claim Boundaries

This directory documents **Active FractalRabbit**, a synthetic-data methods study that
builds an auditable benchmark and claim-discipline framework around sparse-waypoint
active inference. The docs are organized by purpose; start with the orientation below,
then follow the category that matches your task.

> **Scope.** Everything here is bounded to the public NSA-hosted FractalRabbit synthetic
> generator. Nothing in these docs validates real human mobility, implements a formal
> privacy guarantee, or supports deployment. Real-trace claims require the separate
> [empirical-validation protocol](empirical_validation_protocol.md).

## Orientation Files Define the Boundary First

| If you want to… | Read |
| --- | --- |
| Run the pipeline (fixture or external lane) | [`quickstart.md`](quickstart.md) |
| Understand the system end-to-end | [`architecture.md`](architecture.md) |
| Know what is proven vs. explicitly out of scope | [`validation.md`](validation.md) · [`claim_ledger.md`](claim_ledger.md) |
| Check IP, license, citation, and publication status | [`ip_license_status.md`](ip_license_status.md) |

## Methods Documents Specify the Science

| Doc | What it covers |
| --- | --- |
| [`methods/inference-models.md`](methods/inference-models.md) | Discrete POMDP forward filter, the predictor ladder (oracle/hard/soft/soft+boost/AIF), online-adapt, return-boost, precision levers, and the null hierarchy (uniform → base-rate → persistence-oracle). |
| [`methods/fair-frontier-sota.md`](methods/fair-frontier-sota.md) | Why oracle-fed baselines are not the legitimate SOTA target; the fair (oracle-stripped) comparison; the multi-seed provisional verdict; precision self-improvement; and the traveller-count × noise power/regime map. |
| [`methods/statistics.md`](methods/statistics.md) | Per-step vs. traveller-clustered bootstrap CIs (and why the per-step one overstates precision), across-generator-draw seed sweeps, and Brier/ECE calibration with ECE's binning-sensitivity caveat. |
| [`methods/benchmark-protocol.md`](methods/benchmark-protocol.md) | Locked traveller/realization splits, prequential next-cell loss, the false-claim oracle, the `sota_not_cleared` claim gate, and the `eval_contract` guards (G1/G2/G3). |
| [`generative_model_card.md`](generative_model_card.md) | The machine-checkable single-factor and factorial POMDP specification surfaces. |

## Governance Documents Keep Privacy and Dual-Use Claims Bounded

| Doc | What it covers |
| --- | --- |
| [`governance/privacy-and-minimization.md`](governance/privacy-and-minimization.md) | Synthetic re-identification / minimization **proxies** (not formal guarantees), the code-enforced synthetic-provenance gate, governance anchors (GDPR Art. 5(1)(c) / Art. 25, NIST Privacy Framework), and the oversight-positive dual-use boundary. |
| [`empirical_validation_protocol.md`](empirical_validation_protocol.md) | The deliberately separate protocol — consent, provenance, minimization endpoints, privacy review — required before any real-waypoint claim. |

## Publication and Licensing Stay Release-Gated

| Doc | What it covers |
| --- | --- |
| [`ip_license_status.md`](ip_license_status.md) | Private release-candidate posture, MIT code license, CC-BY-4.0 scholarly-output license, third-party simulator/dependency boundaries, and public-release blockers. |

## Reference Documents Keep Interfaces Stable

| Doc | What it covers |
| --- | --- |
| [`config_schema.md`](config_schema.md) | Every supported config key, validation rule, and affected artifact group. |
| [`external_lane.md`](external_lane.md) | The JDK 21 + Maven toolchain for the pinned external FractalRabbit build, with explicit skip reasons when absent. |

## Scholarship Stays Primary-Source Backed

[`scholarship_ledger.md`](scholarship_ledger.md) maps every manuscript citation key to
a primary source and its allowed/forbidden claim language; it is enforced by the
documentation contract.

## Revision Program Retains Historical Planning

The documents in [`revision/`](revision/REVISION_PROGRAM.md) are a frozen planning
record kept for provenance, not current results — do not cite numbers or structure
from them.

- [`revision/REVISION_PROGRAM.md`](revision/REVISION_PROGRAM.md) — master plan (superseded snapshot).
- [`revision/aif-model-designs.md`](revision/aif-model-designs.md) — AIF model design notes.
- [`revision/pass-A-peer-review.md`](revision/pass-A-peer-review.md) — peer-review pass.
- [`revision/pass-B-restructure-imrad.md`](revision/pass-B-restructure-imrad.md) — IMRaD restructure pass.
- [`revision/pass-C-pymdp-push.md`](revision/pass-C-pymdp-push.md) — pymdp push pass.

## Process Documents Preserve Workflows

- Near-term task tracking lives in [`../tasks.yaml`](../tasks.yaml); the
  top-level [`../TODO.md`](../TODO.md) routes only active or blocked work to the
  scoped briefs indexed in [`todo/README.md`](todo/README.md).
- Forward, project-scale work is also scoped in the manuscript roadmap
  ([`../manuscript/07_roadmap.md`](../manuscript/07_roadmap.md)).
- Contributor and agent conventions are in [`AGENTS.md`](AGENTS.md).
