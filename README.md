# RL-Path: AI Agent System for Treatment Optimization

RL-Path is a reinforcement-learning **research simulator** for studying ordered drug-action policies over pathway-level state vectors.

The repository explores a sequential decision problem:

- **state**: simulated pathway activity plus the simulator context needed for future transitions;
- **action**: choose one drug index;
- **transition**: apply immediate/delayed pathway effects, optional pairwise interaction terms, and noise;
- **reward**: reduce simulated disease-pathway activity while penalizing long or broad interventions;
- **agent**: a Deep Q-Network (DQN);
- **baselines**: random and myopic greedy policies.

> **Important:** RL-Path is not a clinical treatment recommendation system. The effect matrix is derived from pathway coverage, the delayed-effect kernel is a modeling choice, and pairwise interaction terms are heuristic simulator parameters. The project does not establish drug efficacy, safety, dosing, or patient benefit.

See [MODEL_CARD.md](MODEL_CARD.md) for scope and limitations.

## Why reinforcement learning?

A static ranking asks which action looks best once. RL-Path instead asks which action to choose **next**, given the current simulated state and recent treatment history.

This matters when the simulator contains delayed effects or interaction terms, because the value of an action can depend on what happened earlier in the episode.

## Environment definition

The default observation contains every variable used by the simulator's transition/reward logic:

```text
current pathway activity
+ disease-pathway mask
+ exact pending delayed-effect schedule
+ recent action history
+ remaining episode fraction
```

The action is one discrete drug index.

The reward is:

```text
reduction in disease-pathway MSE
- per-step penalty
- action-breadth cost
```

This makes the default observation Markov with respect to the implemented simulator.

## Delayed effects

A temporal kernel controls when a simulated pathway effect arrives.

Example:

```text
0.6,0.3,0.1
```

means:

- 60% applies on the current transition;
- 30% applies one transition later;
- 10% applies two transitions later.

The quality-upgraded environment contains an explicit regression test for this timing. A previous implementation decremented delays before checking whether they were due, which caused the first delayed chunk to arrive too early.

## Drug-drug interaction term

The optional interaction tensor modifies the current drug's pathway effect based on recent actions.

When an explicit tensor is not supplied, the repository can construct a simple pathway-overlap heuristic controlled by `--interaction_scale`.

This is a **simulation mechanism**, not a pharmacological interaction database.

## Repository structure

```text
.
├── src/
│   ├── env.py              # state/action/reward simulator
│   ├── dqn.py              # DQN, replay buffer, checkpoints
│   ├── baselines.py        # random + greedy baselines
│   ├── evaluation.py       # paired evaluation + confidence intervals
│   └── preprocess.py       # DGIdb/Reactome effect-matrix construction
├── tests/
├── scripts/
│   ├── PSC_rollout.py
│   ├── psc_pathways_and_drugs.py
│   └── run_seed_sweep.py
├── data/
│   └── README.md
├── train.py
├── evaluate.py
├── EXPERIMENTS.md
├── MODEL_CARD.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml
```

## Installation

```bash
git clone https://github.com/seirana/RL-Path-AI-Agent-System-for-Treatment-Optimization.git
cd RL-Path-AI-Agent-System-for-Treatment-Optimization

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

For tests and linting:

```bash
python -m pip install -e ".[dev]"
```

## Data inputs

The preprocessing workflow expects local research files under `data/raw/` by default:

```text
data/raw/dgidb_interactions.tsv
data/raw/Ensembl2Reactome.txt
```

The public repository does not vendor these external datasets.

The current preprocessing step maps drug-gene interactions to Reactome pathways and creates a normalized drug-to-pathway coverage matrix.

See [data/README.md](data/README.md) for expected inputs and reproducibility notes.

## Train

Example:

```bash
rlpath-train \
  --episodes 400 \
  --steps 10 \
  --top_drugs 60 \
  --top_pathways 40 \
  --temporal_kernel 0.6,0.3,0.1 \
  --interaction_scale 0.15 \
  --interaction_history 1 \
  --eval_rollouts 30 \
  --seed 42
```

Equivalent:

```bash
python train.py ...
```

Training writes a model checkpoint, return/loss traces, an evaluation summary, a learning-curve image, and machine-readable run metadata.

## Evaluate

```bash
rlpath-evaluate \
  --steps 10 \
  --top_drugs 60 \
  --top_pathways 40 \
  --temporal_kernel 0.6,0.3,0.1 \
  --interaction_scale 0.15 \
  --n_rollouts 30 \
  --seed 10000
```

Evaluation compares:

- DQN;
- myopic greedy;
- random.

Each policy receives a **fresh environment with the same rollout seed**. This pairs the disease mask, initial-state draws, and transition-noise stream across policies.

The random policy uses a separate action RNG, so random action selection does not shift the simulator's transition-noise sequence.

The output includes mean, sample standard deviation, standard error, a 95% normal-approximation confidence interval, and paired return differences.

## Independent training seeds

One trained network is not enough to characterize RL optimization variability.

Run several independent training seeds:

```bash
python scripts/run_seed_sweep.py \
  --seeds 11,22,33,44,55 \
  -- \
  --episodes 400 \
  --eval_rollouts 30
```

The sweep stores each run separately and summarizes the DQN evaluation return across training seeds.

See [EXPERIMENTS.md](EXPERIMENTS.md).

## Reproducibility

Randomness is deliberately separated:

- simulator state/noise RNG;
- random-policy action RNG;
- DQN exploration RNG;
- replay-buffer sampling RNG;
- PyTorch initialization seed.

The model checkpoint records:

- observation dimension;
- action count;
- DQN configuration;
- seed;
- model state.

Training also writes `run_metadata.json` with command arguments and software versions.

## Tests

```bash
python -m pytest
```

The suite covers:

- delayed-effect timing;
- Markov observation dimensions;
- independent action/transition RNG streams;
- invalid environment configuration;
- deterministic replay sampling;
- checkpoint shape compatibility;
- paired evaluation statistics;
- effect-matrix serialization without pickled object arrays.

Run lint checks:

```bash
python -m ruff check src tests scripts train.py evaluate.py
```

GitHub Actions runs tests and linting on Python 3.10, 3.11, and 3.12, checks the command-line entry points, and builds the Docker image.

## Docker

Build:

```bash
docker build -t rl-path .
```

A real training run requires the external raw data to be mounted into the container:

```bash
mkdir -p artifacts data/processed

docker run --rm \
  -v "$PWD/data/raw:/app/data/raw:ro" \
  -v "$PWD/data/processed:/app/data/processed" \
  -v "$PWD/artifacts:/app/artifacts" \
  rl-path \
  --episodes 400 \
  --seed 42
```

## PSC-specific scripts

Two PSC-oriented scripts are retained as research demonstrations:

- `scripts/psc_pathways_and_drugs.py` computes pathway-overlap research signals from local files;
- `scripts/PSC_rollout.py` creates a PSC-keyword-matched disease mask and runs a simulated DQN action sequence.

The PSC rollout now aligns the simulator's reward mask with the matched PSC-related pathway subset instead of changing only the initial state.

Outputs from these scripts are **not treatment recommendations**.

## What this project can demonstrate

The repository can support algorithmic questions such as:

- whether DQN learns under the defined simulator;
- whether DQN differs from random/greedy baselines under paired simulated conditions;
- whether learned-policy performance changes under temporal or interaction ablations;
- how much results vary across independent training seeds.

It does not demonstrate clinical effectiveness.

## License

No license file is currently included in this repository. Repository visibility alone does not grant reuse rights; add an explicit license if you intend to permit redistribution or reuse.
