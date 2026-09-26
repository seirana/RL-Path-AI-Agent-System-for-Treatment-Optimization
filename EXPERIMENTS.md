# Experiment protocol

RL-Path is a reinforcement-learning **simulation**. Its drug/pathway effects and
interaction rules are abstractions for studying sequential decision algorithms.
They are not validated pharmacodynamic models and must not be interpreted as
clinical treatment recommendations.

## Experimental questions

The repository is structured to support three comparisons:

1. **Policy comparison** — DQN versus a myopic greedy baseline and a random baseline.
2. **Temporal ablation** — immediate effects versus delayed temporal kernels.
3. **Interaction ablation** — no pairwise interaction term versus positive or negative
   heuristic overlap terms.

## Fair policy evaluation

`src/evaluation.py` evaluates DQN, greedy, and random policies using paired rollout seeds.

For a given rollout seed, each policy gets a fresh environment with the same:

- disease-pathway mask;
- initial-state random draws;
- transition-noise stream;
- environment parameters.

Random action selection uses a separate RNG stream, so sampling a random action does not
shift the transition-noise sequence. This makes the baseline comparison more controlled
than running policies sequentially on one shared RNG.

The output includes mean, sample standard deviation, standard error, and a 95% normal-
approximation confidence interval. It also reports paired return differences for DQN
minus greedy and DQN minus random.

## Independent training seeds

A single trained model is still only one optimization run. For robustness, run several
independent training seeds:

```bash
python scripts/run_seed_sweep.py \
  --seeds 11,22,33,44,55 \
  -- \
  --episodes 400 \
  --eval_rollouts 30
```

Each seed is written to its own artifact directory. The sweep then summarizes the DQN
evaluation-return mean across training seeds.

## Recommended ablation examples

Immediate effects only:

```bash
python train.py \
  --temporal_kernel 1.0 \
  --interaction_scale 0.0 \
  --outdir artifacts/ablation_immediate
```

Delayed effects only:

```bash
python train.py \
  --temporal_kernel 0.6,0.3,0.1 \
  --interaction_scale 0.0 \
  --outdir artifacts/ablation_delayed
```

Delayed effects plus heuristic interaction term:

```bash
python train.py \
  --temporal_kernel 0.6,0.3,0.1 \
  --interaction_scale 0.15 \
  --outdir artifacts/ablation_delayed_interaction
```

These are simulator experiments. A positive `interaction_scale` does not establish
real-world drug synergy, and a negative value does not establish antagonism.

## Reproducibility artifacts

Every training run writes:

- `dqn.pt` — model checkpoint with architecture/configuration metadata;
- `metrics.json` — training and paired-evaluation metrics;
- `eval_summary.json` — policy summaries and confidence intervals;
- `returns.json` — per-episode returns;
- `losses.json` — optimization losses;
- `learning_curve.png` — training-return curve;
- `run_metadata.json` — command arguments, Python/platform information, NumPy/Torch
  versions, checkpoint path, and effect-matrix path.

Generated artifacts are excluded from Git by default.

## Interpretation

The strongest conclusions supported by these experiments are algorithmic:

- whether a policy learns under the defined simulator;
- whether it outperforms the included baselines under paired simulated conditions;
- how performance changes when the simulator's temporal/interaction assumptions change.

The experiments do **not** establish drug efficacy, safety, dosing, sequence validity,
or patient-level treatment benefit.
