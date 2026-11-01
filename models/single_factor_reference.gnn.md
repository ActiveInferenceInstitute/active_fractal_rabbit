# GNN spec: Active Fractal Rabbit single-factor reference model

## GNNSection
ActInfAFRSingleFactor

## GNNVersionAndFlags
GNN v1.1

## ModelName
Active Fractal Rabbit Single-Factor Reference

## ModelAnnotation
Synthetic-only generative-model specification for the single-factor reference
model of the fixture-first lane. One hidden factor (the discretized spatial
cell) and four observation modalities; every modality depends on the cell
alone, so the joint likelihood uses the conditional-independence approximation
across modalities.

This file is the source of truth for model *structure* only. The A/B/D tensors
are learned from declared training rows by
`src/active_fractal_rabbit/pomdp.py`; they are deliberately not fixed numeric
constants, so no `## InitialParameterization` block is declared. The default
4x4 fixture surface is 16 spatial cells, 4 time bins, 4 speed categories, and
2 burst categories.

Spec estimation contract: A/B/D estimated from declared training rows only; smoothing is explicit; evaluation-holdout rows do not update tensors
Spec tuning contract: precision/return selected after inner-train transition fitting; online adaptation selected in a second predeclared inner-validation stage
Spec notes: Single hidden factor (spatial cell). Each A_m is column-stochastic of shape (obs_size, num_states); the joint likelihood P(o|s) = prod_m A_m[o_m, s] uses the conditional-independence approximation across modalities. B is (S, S, n_controls), each control slice column-stochastic; D is a length-S simplex.

## StateSpaceBlock
# Hidden state factor (one factor): discretized spatial cell.
s_cell[16,1,type=float]              # hidden factor 0: 4x4 spatial cell surface

# Observation modalities (four), one variable per modality.
o_location[16,1,type=int]            # modality 0: observed cell
o_time_of_day[4,1,type=int]          # modality 1: time-of-day bin
o_speed[4,1,type=int]                # modality 2: speed category
o_report_burst[2,1,type=int]         # modality 3: sparse-report burst indicator

# Likelihood (emission) matrices, one per modality: A[obs_size, factor_size].
A_location[16,16,type=float]
A_time_of_day[4,16,type=float]
A_speed[4,16,type=float]
A_report_burst[2,16,type=float]

# Transition matrix for the single factor: B[size, size, n_controls].
B_cell[16,16,2,type=float]

# Log-preference vectors, one per modality.
C_location[16,type=float]
C_time_of_day[4,type=float]
C_speed[4,type=float]
C_report_burst[2,type=float]

# Initial-state prior for the single factor.
D_cell[16,type=float,default=uniform]

# Controls for the single controllable factor.
u_cell[2,1,type=int]                 # discrete control index (2 controls)
π_cell[2,type=float]                 # policy posterior over the 2 controls

# Free energy and time.
F[1,type=float]
G[1,type=float]
t[1,type=int]

## Connections
# Priors initialize the hidden factor.
D_cell>s_cell

# Likelihood dependencies: every modality depends on the cell factor alone.
s_cell>A_location
s_cell>A_time_of_day
s_cell>A_speed
s_cell>A_report_burst

# Emissions produce observations.
A_location>o_location
A_time_of_day>o_time_of_day
A_speed>o_speed
A_report_burst>o_report_burst

# Transitions depend on the cell factor and its control.
s_cell>B_cell
u_cell>B_cell
B_cell>s_cell

# Preferences and policy selection.
C_location-o_location
C_time_of_day-o_time_of_day
C_speed-o_speed
C_report_burst-o_report_burst
G>π_cell
π_cell>u_cell

## Equations
# Conditional-independence likelihood:
#   P(o | s) = prod_m A_m[o_m, s]
# Column-stochastic emission and transition normalization:
#   sum_o A_m[o, s] = 1 for every state s
#   sum_{s'} B[s', s, a] = 1 for every state s and control a
# A/B/D values are estimated from declared training rows, not fixed here.

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=30

## ActInfOntologyAnnotation
s_cell=HiddenStateFactor0
o_location=ObservationModality0
o_time_of_day=ObservationModality1
o_speed=ObservationModality2
o_report_burst=Observation
A_location=LikelihoodMatrixModality0
A_time_of_day=LikelihoodMatrixModality1
A_speed=LikelihoodMatrixModality2
A_report_burst=LikelihoodMatrix
B_cell=TransitionMatrixFactor0
C_location=LogPreferenceVectorModality0
C_time_of_day=LogPreferenceVectorModality1
C_speed=LogPreferenceVectorModality2
C_report_burst=Preferences
D_cell=PriorOverHiddenStatesFactor0
u_cell=Action
π_cell=PolicyVector
t=Time

## ModelParameters
model_id: single_factor_reference
num_factors: 1
num_hidden_states: 16
num_obs: 26
num_modalities: 4
num_actions: 2
num_timesteps: 30
conditional_independence: true
num_hidden_states_factor0: 16
num_obs_modality0: 16
num_obs_modality1: 4
num_obs_modality2: 4
num_obs_modality3: 2

## Footer
Active Fractal Rabbit Single-Factor Reference — GNN v1.1 structural spec.
Synthetic-only; A/B/D are learned from declared training rows and are not fixed
numeric constants in this file.

## Signature
Structural spec only; the upstream GNN revision is pinned in
`.github/gnn-pin.json` and in `models/README.md`.
