# Model and simulator card

## Intended use

RL-Path is a research project for studying reinforcement-learning behavior in a
pathway-steering simulation. It is suitable for software-engineering, RL-method,
and reproducibility experiments.

It is not intended for clinical decision support, prescribing, dosing, patient
stratification, or real-world treatment-sequence selection.

## Agent

The current agent is a Deep Q-Network with:

- two hidden fully connected layers;
- replay-buffer training;
- epsilon-greedy exploration;
- a target network;
- gradient clipping.

The checkpoint stores the observation dimension, action count, training seed, and
DQN configuration together with the learned state dictionary.

## Simulator state

By default, the observation contains:

- current pathway activity;
- the disease-pathway mask used by the reward;
- the exact future delayed-effect schedule;
- recent actions required by the pairwise interaction rule;
- the remaining episode fraction.

These variables are included because they influence future transition or reward
behavior in the implemented simulator.

## Actions

Each action is a discrete drug label from the processed effect matrix.

The presence of a drug in the action space does not mean it is appropriate, safe,
effective, or indicated for any disease.

## Reward

The reward is based on change in simulated disease-pathway mean-squared activity,
with penalties for each step and for broad pathway effects.

The reward is an engineering objective. It is not a validated clinical endpoint.

## Drug-to-pathway matrix

The preprocessing code maps DGIdb drug-gene records to Reactome pathways and turns
pathway coverage counts into normalized per-drug vectors.

This representation does not encode dose, direction of regulation, tissue context,
exposure, pharmacokinetics, adverse effects, clinical evidence, or patient-specific
biology.

## Delayed effects

The temporal kernel is supplied by the experimenter. For example,
`0.6,0.3,0.1` distributes an effect across three simulated transitions.

The kernel is not estimated from pharmacokinetic or pharmacodynamic data.

## Pairwise interactions

If an explicit interaction tensor is not provided, the simulator constructs a
pathway-overlap heuristic scaled by `interaction_scale`.

Positive scale values should be interpreted only as positive simulator interaction
terms. They do not establish drug synergy. Negative values do not establish
antagonism.

## Evaluation

DQN is compared with random and myopic greedy policies on paired rollout seeds.

Reported confidence intervals characterize variability in the simulator rollouts.
They do not represent confidence intervals for clinical outcomes.

Independent training-seed sweeps are supported because optimization variance can
be substantial in reinforcement learning.

## Known limitations

- no patient-level state;
- no dose or schedule units;
- no adverse-event model;
- no pharmacokinetic/pharmacodynamic model;
- no causal treatment-effect identification;
- no clinical-outcome labels;
- no validated drug-drug interaction database in the default simulator;
- no external or prospective clinical validation;
- preprocessing depends on external database contents and identifier mapping;
- DQN performance can be sensitive to hyperparameters and random seed.

## Appropriate claims

Appropriate:

> Under the specified simulator and evaluation seeds, the learned DQN achieved the
> reported return distribution relative to the included baselines.

Not appropriate:

> The model found an effective treatment sequence for a patient or disease.

## Reproducibility

For any reported experiment, preserve:

- Git commit SHA;
- exact external data versions and retrieval dates;
- preprocessing outputs or checksums;
- CLI arguments;
- training seed(s);
- evaluation seed range;
- software versions;
- generated `run_metadata.json`, `metrics.json`, and `eval_summary.json`.
