# Supplementary S3 - FractalRabbit Becomes Categorical Evidence, Not Real Mobility {#sec:simulator-boundary}

This supplement documents the exogenous FractalRabbit generator contract and the observation-discretization pipeline that converts raw waypoints into the categorical evidence consumed by the main-line inference layer ([@sec:methodology]).

## The Pinned Upstream Contract Defines the Fixture Lane {#sec:simulator-pinned-contract}

FractalRabbit is treated as an upstream stochastic data generator, not as project-local source. The pinned external lane writes the published parameter rows with labels matching `parameters.csv`, invokes the built jar as `java -cp fractalrabbit-1.0-jar-with-dependencies.jar fractalRabbitGenerator.MainClassFR parameters.csv <output_prefix>` (the simulator appends `XY.csv` and `PLACES.csv` to the prefix rather than taking an output filename), and parses only the documented output record shape [@nsa2026fractalrabbit; @nsa2026parameters; @nsa2026mainclass]. Because the upstream repository currently has no release archive, the manuscript cites commit-specific URLs and stores the exact commit in the generated provenance manifest, following software-citation principles of specificity and persistence [@smith2016software_citation]. The fixture lane uses a checked CSV with the same columns, allowing Python tests and manuscript generation to run when Java and Maven are unavailable. The boundary layer resolves a JDK 21 runtime explicitly (environment override, PATH, Homebrew keg locations, then `java_home`) and exports `JAVA_HOME` to the Maven and Java subprocesses, because Homebrew installs `openjdk@21` keg-only. On the current host the external lane status is: {{EXTERNAL_RUN_STATUS}} — an artifact-presence status derived from the retained pinned-build metrics file, not a statement that the lane completed within the current full-study invocation (the current-run gate artifacts report their own external-lane status independently).

## Fractal Dimension Is a Generator Knob, Not a Real-Trajectory Estimate {#sec:fractal-dimension-control}

The simulator-facing parameter vector is:

$$
\theta_{\mathrm{FR}}
= (d, h, N_{\mathrm{pts}}, N_{\mathrm{trav}}, c, T_{\mathrm{days}}, N_{\mathrm{rep}}),
$$ {#eq:fractalrabbit-params}

where $d$ is the ambient space dimension, $h$ the fractal dimension, $N_{\mathrm{pts}}$ the generated point count, $N_{\mathrm{trav}}$ the traveller count, $c$ the co-traveller count, $T_{\mathrm{days}}$ the duration in days, and $N_{\mathrm{rep}}$ the mean report count. [@eq:fractalrabbit-params] is a wrapper around the upstream CSV surface; it is not a fitted statistical parameter vector in this project.

The dimension parameter obeys the generator-side range:

$$
0 < h \leq d,
$$ {#eq:fractal-dimension-range}

with this scaffold restricted to the upstream two-dimensional waypoint output. We use $h$ as a controlled synthetic roughness knob: lower values concentrate generated support into a thinner set of places, while higher values spread the support through more of the ambient plane. Movement-ecology and human-mobility models make clear why this distinction matters: generator controls, individual movement mechanisms, and observation processes are different objects even when their outputs can be plotted as trajectories [@nathan2008movement_ecology; @barbosa2018human_mobility]. The project therefore asks how downstream inference diagnostics respond when $h$ changes; it does not estimate a box-counting, Hausdorff, or correlation dimension from observed traces, and it does not claim that any generated $h$ matches real movement.

## CSV Parameters Expose Only Part of the Upstream Simulator {#sec:simulator-exposed-fixed}

The pinned upstream source implements three explicit tiers, each with its own
parameters. The retro-preferential tier is useful benchmark context for exploration-and-return mobility ideas, but this project treats it as a simulator boundary rather than a fitted empirical law [@song2010scaling; @darling2018fractalrabbit]. The CSV interface exposes seven of these parameters; the remaining internals are
fixed constants in the upstream main class, documented here so the configurability
boundary is exact rather than implied:

| tier (upstream class) | role | CSV-exposed parameters | upstream-fixed internals |
| --- | --- | --- | --- |
| `AgoraphobicPoints` | heavy-tailed fractal place process | space dimension $d$, fractal dimension $h$, point count $N_{\mathrm{pts}}$ | clump restart rate $\theta = 0.75$ |
| `Retropreferential` | retro-preferential trajectory process over places | traveller count $N_{\mathrm{trav}}$, co-traveller count $c$ | distance exponent $-2.0$, exploration mean $\phi = 10$, trajectory-length mean $100$ |
| `SporadicReporter` | sporadic report-time process along trajectories | duration $T_{\mathrm{days}}$, mean report count $N_{\mathrm{rep}}$ | speed bound $50$ units/day, $200$ km/unit, Pareto cutoff $0.001$, tail $-1.5$ |

Every CSV-exposed parameter is configurable through `fractalrabbit.params`; the
upstream-fixed internals cannot be changed without forking the pinned commit, and
this project deliberately does not fork it. The simulator emits two row-aligned
output streams per run: the waypoint stream `*XY.csv` consumed as observations,
and the ground-truth stream `*PLACES.csv` (`Traveler ID,Days,Place ID`) naming
the true `AgoraphobicPoints` place behind every report. The pipeline ingests
both: places feed the revisit-structure statistics and the ground-truth state
recovery evaluation in [@sec:results].

## The Parser Consumes a Narrow Waypoint Record Contract {#sec:simulator-waypoint-contract}

The parser accepts rows of the form:

$$
w_i = (\mathrm{id}_i, t_i, x_i, y_i),
$$ {#eq:waypoint-record}

with $t_i$ measured in days and positions measured in kilometers. The parser sorts by traveller and time and writes `output/runs/*/observations.csv` as the raw sorted post-parser artifact; the downstream observation-discretization step ([@sec:observation-model]) subsequently validates nonnegative time before any categorical features are derived.

## The Observation Model Converts Waypoints into Finite Evidence {#sec:observation-model}

With the simulator boundary fixed, the observation layer turns the raw waypoint stream above into the finite categorical evidence the generative model of [@sec:formal-model] consumes.

### Categorical Features Bound the Evidence Surface {#sec:observation-categorical-surface}

The observation layer maps continuous, irregular waypoint records into finite categorical modalities. The default emitted feature table has columns {{DISCRETE_FEATURE_COLUMNS}} and is the source for the spatial, temporal, speed, and burst figures in [@sec:results].

### Spatial States Are Analysis Cells, Not Simulator Internals {#sec:observation-spatial-state}

For each sorted waypoint, spatial discretization maps coordinates into a finite cell:

$$
s_i = b_y(y_i) \cdot G + b_x(x_i),
$$ {#eq:spatial-state}

where $G$ is the configured spatial bin count per axis and $b_x, b_y$ are the configured spatial binning functions. The current implementation supports equal-width grids, fixed-bound grids, and empirical quantile bins. The latent state in [@eq:spatial-state] is a categorical analysis state, not the hidden point identity inside FractalRabbit; because location is also emitted as an observation modality, the default fixture is close to a directly observed-state filter rather than a deep latent-uncertainty benchmark.

### Temporal, Kinematic, and Burst Features Remain Heuristic Cues {#sec:observation-temporal-kinematic}

Time-of-day is represented as:

$$
b_t:[0,1)\to\{0,\ldots,G_t-1\},
\qquad
\tau_i = b_t(t_i \bmod 1),
$$ {#eq:time-bin}

where $G_t$ is the configured number of time-of-day bins and $b_t$ is the explicit temporal binning map. Within each traveller, inter-report interval and speed are:

$$
\Delta t_i = t_i - t_{i-1},
\qquad
v_i =
\begin{cases}
\dfrac{\sqrt{(x_i-x_{i-1})^2 + (y_i-y_{i-1})^2}}{\Delta t_i} & \Delta t_i > 0, \\[4pt]
0 & \Delta t_i \le 0.
\end{cases}
$$ {#eq:interval-speed}

Non-positive intervals are not floored to a positive denominator; the implementation instead leaves $v_i$ at its zero-initialized sentinel value, distinct from the observation-noise probability in [@sec:formal-partial-observability]. The first waypoint for each traveller has no previous within-traveller report, so the emitted feature table marks {{FIRST_OBSERVATION_COUNT}} rows with `is_first_observation=1` and assigns the {{FIRST_OBSERVATION_HANDLING}}. This sentinel convention keeps all waypoint rows in the categorical summaries while avoiding an artificial positive interval.

Speed categories are thresholded from $v_i$, while burst categories are assigned by the configured interval rule. In the default run, the burst rule is `{{BURST_RULE}}`, so short intervals are defined relative to the positive-interval median rather than a universal biological or operational threshold. We use "burst" in the temporal-dynamics sense of clustered reporting separated by longer waiting times; the current median-positive rule is a heuristic discretizer, not a full burstiness statistic, and finite event sequences require particular caution before treating a burst score as stable [@barabasi2005bursts; @goh2008burstiness; @kim2016burstiness_finite].

### Fractal-Dimension Sweeps Shift Occupancy Through the Same Discretizer {#sec:observation-fractal-implications}

The sweep introduced in [@sec:fractal-dimension-control] changes the upstream support before this observation layer sees any rows. The downstream state count, occupancy profile, revisit rate, and predictive losses are therefore consequences of the same discretizer applied to generator outputs with different support roughness. This is why the sweep is reported as generator sensitivity rather than a direct fractal-dimension estimator.
