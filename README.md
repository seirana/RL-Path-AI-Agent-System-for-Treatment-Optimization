# RL-Path: AI Agent System for Treatment Optimization

RL-Path is a bioinformatics reinforcement learning project that learns **ordered drug interventions** to steer a simulated disease state toward a healthier pathway profile.

This version frames the repository as an **AI agent system for treatment optimization**:
- **Agent**: a DQN policy that chooses the next drug
- **Environment**: a pathway-steering simulator built from drug→pathway effects
- **Objective**: reduce disease-associated pathway activity while penalizing long or broad interventions

Compared with the original lightweight baseline, this package adds two important modeling extensions:
- **Delayed drug effects** via a temporal kernel
- **Drug–drug interactions** via pairwise pathway-level interaction effects

These changes make the treatment dynamics more biologically realistic and make the observation more appropriate for DQN by exposing the parts of treatment history that matter for future transitions.

---

## Why this is an AI agent system

This repository does not just rank drugs statically. It learns a **state-dependent treatment policy**.

At each step, the agent:
1. observes the current pathway state
2. selects a drug action
3. receives feedback based on how much disease-pathway activity is reduced
4. updates its policy from replayed experience

That makes the project a true **sequential decision-making system** for treatment optimization.

---

## What changed in this version

### 1. Markov-aware observation design
The observation now includes:
- current pathway activity vector
- summary of pending delayed effects still scheduled to land in future steps
- one-hot encoding of the most recent drug action
- remaining-step fraction

This matters because once delayed effects or interaction history are added, the pathway vector alone is no longer enough for a value-based RL agent.

### 2. Delayed effects
Use `--temporal_kernel` to control how a drug acts over time.

Example:
```bash
--temporal_kernel 0.6,0.3,0.1
```
This means 60% of the effect lands immediately, 30% on the next step, and 10% one step later.

### 3. Drug–drug interactions
Use `--interaction_scale` to enable a simple heuristic interaction tensor derived from overlap in pathway coverage.

- positive values: synergy
- negative values: antagonism

By default, the interaction is applied between the current drug and the most recent prior drug, but you can enlarge that window with `--interaction_history`.

---

## Repository structure

```text
RL-Path/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  ├─ __init__.py
│  ├─ preprocess.py
│  ├─ env.py
│  ├─ dqn.py
│  └─ baselines.py
├─ train.py
├─ evaluate.py
├─ scripts/
│  ├─ psc_pathways_and_drugs.py
│  └─ psc_rollout.py
├─ data/
│  ├─ raw/
│  └─ processed/
└─ artifacts/
```

---

## Data sources

DGIdb interactions:
```text
https://www.dgidb.org/data/latest/interactions.tsv
```

Reactome mapping:
```text
https://download.reactome.org/current/Ensembl2Reactome.txt
```

Place them under:
```text
data/raw/dgidb_interactions.tsv
data/raw/Ensembl2Reactome.txt
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Train

Baseline training with delayed effects and pairwise interactions:

```bash
python train.py \
  --episodes 400 \
  --steps 10 \
  --top_drugs 60 \
  --top_pathways 40 \
  --temporal_kernel 0.6,0.3,0.1 \
  --interaction_scale 0.15 \
  --interaction_history 1
```

To recover the old immediate-effect behavior:

```bash
python train.py --temporal_kernel 1.0 --interaction_scale 0.0
```

---

## Evaluate

```bash
python evaluate.py \
  --steps 10 \
  --top_drugs 60 \
  --top_pathways 40 \
  --temporal_kernel 0.6,0.3,0.1 \
  --interaction_scale 0.15
```

---

## Output artifacts

Training and evaluation write outputs into `artifacts/`, for example:
- `dqn.pt`
- `metrics.json`
- `returns.json`
- `losses.json`
- `learning_curve.png`
- `eval_summary.json`

---

## Notes on implementation

- `train.py` no longer uses a hard-coded absolute raw-data path; it now defaults to `data/raw`.
- `env.py` exposes `obs_dim`, which should be used by the agent instead of `n_pathways`.
- `scripts/psc_rollout.py` was updated so it works with the expanded observation and corrected processed-data path.
- The greedy baseline was adapted to the richer environment logic.

---

## Suggested experiments

1. **Ablation study**
   - immediate effects only
   - delayed effects only
   - delayed effects + interactions

2. **Interaction sign study**
   - positive interaction scale for synergy
   - negative interaction scale for antagonism

3. **Temporal sensitivity**
   - compare short kernels like `1.0`
   - with broader kernels like `0.5,0.3,0.2`

These comparisons help show whether sequence-aware treatment control becomes more valuable once treatment history matters.
