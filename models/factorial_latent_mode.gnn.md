# GNN spec: Active Fractal Rabbit factorial latent-mode model

## GNNSection
ActInfAFRFactorialLatentMode

## GNNVersionAndFlags
GNN v1.1

## ModelName
Active Fractal Rabbit Factorial Latent-Mode Model

## ModelAnnotation
Synthetic-only generative-model specification for the three-factor latent-mode
model of the fixture-first lane. Three hidden factors (`cell`,
`movement_mode`, `cognitive_proxy_mode`) and six observation modalities. The
likelihood and transition dependencies couple factors, so the exact posterior
is non-factorized and mean-field inference is an approximation reported with
its gap.

This file is the source of truth for model *structure* only. The A/B/D tensors
are learned from declared included rows by
`src/active_fractal_rabbit/factorial_pymdp.py`; they are deliberately not fixed
numeric constants, so no `## InitialParameterization` block is declared. The
default 4x4 fixture surface is 16 spatial cells with three-state movement and
cognitive-proxy modes.

Spec estimation contract: factorial A/B/D tensors estimated from declared included rows; the current factorial surface is diagnostic, not a locked predictive candidate
Spec tuning contract: no predictive hyperparameter promotion; tensor and mean-field diagnostics only
Spec notes: Three hidden factors. A_dependencies couple factors (e.g. time_of_day depends on cell and cognitive-proxy), so the exact posterior is non-factorized and mean-field inference is an approximation reported with its gap. Each A_m is column-stochastic over its observation axis; each B factor tensor is column-stochastic over its target axis; each D factor is a simplex. The cognitive-proxy factor is a synthetic reporting-process state, not real cognition or intent.

## StateSpaceBlock
# Hidden state factors (three): cell, movement mode, cognitive-proxy mode.
s_cell[16,1,type=float]                  # factor 0: 4x4 spatial cell surface
s_movement_mode[3,1,type=float]          # factor 1: latent movement mode
s_cognitive_proxy_mode[3,1,type=float]   # factor 2: synthetic reporting-process mode

# Observation modalities (six), one variable per modality.
o_location[16,1,type=int]                # modality 0: observed cell
o_time_of_day[4,1,type=int]              # modality 1: time-of-day bin
o_speed[4,1,type=int]                    # modality 2: speed category
o_report_burst[2,1,type=int]             # modality 3: sparse-report burst indicator
o_cell_delta[2,1,type=int]               # modality 4: cell-change indicator
o_return_flag[2,1,type=int]              # modality 5: return-to-cell indicator

# Likelihood (emission) matrices: A[obs_size, factor_size, ...] in dependency order.
A_location[16,16,type=float]
A_time_of_day[4,16,3,type=float]
A_speed[4,3,16,type=float]
A_report_burst[2,3,type=float]
A_cell_delta[2,3,type=float]
A_return_flag[2,3,type=float]

# Transition matrices, one per factor: B[size, size, n_controls].
B_cell[16,16,2,type=float]
B_movement_mode[3,3,2,type=float]
B_cognitive_proxy_mode[3,3,2,type=float]

# Log-preference vectors, one per modality.
C_location[16,type=float]
C_time_of_day[4,type=float]
C_speed[4,type=float]
C_report_burst[2,type=float]
C_cell_delta[2,type=float]
C_return_flag[2,type=float]

# Initial-state priors, one per factor.
D_cell[16,type=float,default=uniform]
D_movement_mode[3,type=float,default=uniform]
D_cognitive_proxy_mode[3,type=float,default=uniform]

# Controls, one set per controllable factor.
u_cell[2,1,type=int]
u_movement_mode[2,1,type=int]
u_cognitive_proxy_mode[2,1,type=int]
π_cell[2,type=float]
π_movement_mode[2,type=float]
π_cognitive_proxy_mode[2,type=float]

# Free energy and time.
F[1,type=float]
G[1,type=float]
t[1,type=int]

## Connections
# Priors initialize the hidden factors.
D_cell>s_cell
D_movement_mode>s_movement_mode
D_cognitive_proxy_mode>s_cognitive_proxy_mode

# Likelihood dependencies: edge order defines each A dependency tuple.
s_cell>A_location
s_cell>A_time_of_day
s_cognitive_proxy_mode>A_time_of_day
s_movement_mode>A_speed
s_cell>A_speed
s_cognitive_proxy_mode>A_report_burst
s_movement_mode>A_cell_delta
s_movement_mode>A_return_flag

# Emissions produce observations.
A_location>o_location
A_time_of_day>o_time_of_day
A_speed>o_speed
A_report_burst>o_report_burst
A_cell_delta>o_cell_delta
A_return_flag>o_return_flag

# Transitions depend on coupled factors and their controls.
s_cell>B_cell
s_movement_mode>B_cell
u_cell>B_cell
B_cell>s_cell
s_movement_mode>B_movement_mode
s_cognitive_proxy_mode>B_movement_mode
u_movement_mode>B_movement_mode
B_movement_mode>s_movement_mode
s_cognitive_proxy_mode>B_cognitive_proxy_mode
u_cognitive_proxy_mode>B_cognitive_proxy_mode
B_cognitive_proxy_mode>s_cognitive_proxy_mode

# Preferences and policy selection.
C_location-o_location
C_time_of_day-o_time_of_day
C_speed-o_speed
C_report_burst-o_report_burst
C_cell_delta-o_cell_delta
C_return_flag-o_return_flag
G>π_cell
π_cell>u_cell
G>π_movement_mode
π_movement_mode>u_movement_mode
G>π_cognitive_proxy_mode
π_cognitive_proxy_mode>u_cognitive_proxy_mode

## Equations
# Coupled likelihood: each A_m is column-stochastic over its observation axis
# and depends on the factor set named by its dependency tuple.
#   sum_o A_m[o, ...] = 1 for every conditioning slice
# Coupled transitions: each B factor tensor is column-stochastic over its
# target axis and conditions on the factor set named by its dependency tuple.
#   sum_{s'} B_f[s', ...] = 1 for every conditioning slice
# The exact posterior is non-factorized; mean-field inference is approximate.
# A/B/D values are estimated from declared included rows, not fixed here.

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=30

## ActInfOntologyAnnotation
s_cell=HiddenStateFactor0
s_movement_mode=HiddenStateFactor1
s_cognitive_proxy_mode=HiddenState
o_location=ObservationModality0
o_time_of_day=ObservationModality1
o_speed=ObservationModality2
o_report_burst=ObservationModality3
o_cell_delta=ObservationModality4
o_return_flag=ObservationModality5
A_location=LikelihoodMatrixModality0
A_time_of_day=LikelihoodMatrixModality1
A_speed=LikelihoodMatrixModality2
A_report_burst=LikelihoodMatrixModality3
A_cell_delta=LikelihoodMatrixModality4
A_return_flag=LikelihoodMatrixModality5
B_cell=TransitionMatrixFactor0
B_movement_mode=TransitionMatrixFactor1
B_cognitive_proxy_mode=TransitionMatrix
C_location=LogPreferenceVectorModality0
C_time_of_day=LogPreferenceVectorModality1
C_speed=LogPreferenceVectorModality2
C_report_burst=LogPreferenceVectorModality3
C_cell_delta=LogPreferenceVectorModality4
C_return_flag=LogPreferenceVectorModality5
D_cell=PriorOverHiddenStatesFactor0
D_movement_mode=PriorOverHiddenStatesFactor1
D_cognitive_proxy_mode=PriorOverHiddenStates
u_cell=Action
u_movement_mode=Action
u_cognitive_proxy_mode=Action
π_cell=PolicyVector
π_movement_mode=PolicyVectorFactor1
π_cognitive_proxy_mode=PolicyVector
t=Time

## ModelParameters
model_id: factorial_latent_mode
num_factors: 3
num_hidden_states: 22
num_obs: 30
num_modalities: 6
num_actions: 2
num_controls_per_factor: 2
num_timesteps: 30
conditional_independence: false
num_hidden_states_factor0: 16
num_hidden_states_factor1: 3
num_hidden_states_factor2: 3
num_obs_modality0: 16
num_obs_modality1: 4
num_obs_modality2: 4
num_obs_modality3: 2
num_obs_modality4: 2
num_obs_modality5: 2

## Footer
Active Fractal Rabbit Factorial Latent-Mode Model — GNN v1.1 structural spec.
Synthetic-only; A/B/D are learned from declared included rows and are not fixed
numeric constants in this file. The cognitive-proxy factor is a synthetic
reporting-process state, not real cognition or intent.

## Signature
Structural spec only; the upstream GNN revision is pinned in
`.github/gnn-pin.json` and in `models/README.md`.
