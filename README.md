# ATHENA — Adaptive Trajectory-Hierarchical Episodic Neural Architecture

A continual-learning research framework for **sequential multi-domain language-model
training without catastrophic forgetting**. ATHENA splits a LoRA-adapted LLM into a
fast (plastic) pathway and a slow (stable) pathway, and governs their interaction
using biologically-inspired memory mechanisms.

This repository currently contains **Step 1 (data pipeline)** and **Step 2 (math
core)** of the build.

---

## Table of Contents

1. [Concept / Architecture](#1-concept--architecture)
2. [Repository Structure](#2-repository-structure)
3. [The Three Pillars](#3-the-three-pillars)
4. [Mathematical Specification](#4-mathematical-specification)
5. [Hyperparameters](#5-hyperparameters)
6. [Data Pipeline (Step 1)](#6-data-pipeline-step-1)
7. [Math Core (Step 2)](#7-math-core-step-2)
8. [Setup & Installation](#8-setup--installation)
9. [Running the Code](#9-running-the-code)
10. [Verification & Test Results](#10-verification--test-results)
11. [Known Environment Notes](#11-known-environment-notes)
12. [Roadmap](#12-roadmap)

---

## 1. Concept / Architecture

Standard fine-tuning overwrites old knowledge when trained on a new task
(*catastrophic forgetting*). ATHENA addresses this with a **dual-pathway** design:

```
                    ┌──────────────────────────────────────────────┐
                    │            Incoming Task Batch                │
                    │       (input_ids, attention_mask, labels)     │
                    └───────────────────────┬──────────────────────┘
                                            │
                       ┌────────────────────▼────────────────────┐
                       │  PILLAR 1 · Cognitive Selection Gate     │
                       │  surprise = per-token NLL over VALID      │
                       │  tokens (-100 padding excluded)           │
                       └────────────────────┬────────────────────┘
                          surprise > τ ?     │
                   ┌───────────────────────┴───────────────────────┐
             NO (familiar)                                 YES (novel / at-risk)
                   │                                              │
                   ▼                                              ▼
        normal training only                     ┌───────────────────────────────┐
                                                 │  Cognitive Replay Buffer       │
                                                 │  · admission by surprise       │
                                                 │  · Ebbinghaus decay tracking   │
                                                 │  · capacity eviction (≤400)    │
                                                 │  · At-Risk rescue replays      │
                                                 └──────────────┬────────────────┘
                                                                │ replayed batch
                                                                ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │                          PILLAR 2 · Parameter Dynamics                        │
   │   Fast (Plastic) LoRA   θ_F   ── evolves every step ──▶  trajectory d_θ        │
   │                                            velocity v_t  ──▶  Instability r   │
   └───────────────────────────────────────┬──────────────────────────────────────┘
                                           │ d_θ > Δ_θ_threshold ?
                                           ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │              PILLAR 3 · Instability-Gated EMA Consolidation                   │
   │   Slow (Stable) LoRA  θ_S  ◀── θ_S ← (1−α_eff)·θ_S + α_eff·θ_F                │
   │   α_eff = α_base / (1 + γ·r)   (shields slow pathway when θ_F is volatile)  │
   └────────────────────────────────────────────────────────────────────────────┘
```

The **slow pathway** is the long-term memory that receives consolidated knowledge;
the **fast pathway** is the short-term learner. Consolidation only happens when the
fast pathway has moved far enough from its last reference snapshot (*"model time"*),
and the transfer rate shrinks automatically when the fast pathway is unstable.

---

## 2. Repository Structure

```
athena_project/
│
├── config.yaml                  # Global config + ATHENA hyperparameters
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignores venvs & __pycache__ (data/ IS tracked)
├── README.md                    # This file
│
├── data/
│   └── processed/               # Pre-tokenized Arrow datasets (Step 1 output)
│       ├── task_1_medical/
│       │   ├── train/           # data-00000-of-00001.arrow + dataset_info/state.json
│       │   └── val/
│       ├── task_2_legal_finance/
│       │   ├── train/
│       │   └── val/
│       └── task_3_superglue_boolq/
│           ├── train/
│           └── val/
│
└── src/
    ├── data_pipeline.py         # Step 1: download → format → tokenize → save
    ├── verify_data.py           # Step 1: validate Arrow datasets + label masking
    ├── cognitive_gate.py        # Step 2, Pillar 1: gate + replay buffer
    ├── dynamics_and_ema.py      # Step 2, Pillars 2 & 3: dynamics + EMA consolidation
    └── test_step2_math.py       # Step 2: full CPU verification suite (3 tests)
```

### Per-file responsibilities

| File | Purpose | Key public API |
|---|---|---|
| `src/data_pipeline.py` | Ingests three public datasets, converts each row to a unified `Instruction/Context/Response` prompt, tokenizes to length 512, stores as Arrow | `main()`, `TASKS`, `FORMATTERS` |
| `src/verify_data.py` | Re-opens every saved dataset, asserts shapes and the `-100` padding mask invariant | `main()` |
| `src/cognitive_gate.py` | Pillar 1 — NLL surprise gate and the dual-criteria replay buffer | `cognitive_selection_gate()`, `CognitiveReplayBuffer` |
| `src/dynamics_and_ema.py` | Pillars 2 & 3 — trajectory distance, instability ratio, EMA consolidation | `compute_trajectory_distance()`, `ParameterDynamicsAndConsolidation` |
| `src/test_step2_math.py` | End-to-end CPU test of Pillars 1–3 with printable summary tables | `main()` |

---

## 3. The Three Pillars

### Pillar 1 — Cognitive-Aware Selection Gate & Replay Buffer

* **Selection gate** — scores each incoming sample by its *Negative Log-Likelihood
  Surprise* (mean NLL over valid tokens only). Samples above `surprise_threshold`
  are "novel / at-risk" and are admitted to the buffer.
* **Replay buffer** — stores admitted samples and tracks their *Memory Strength*
  using the biological **Ebbinghaus forgetting curve**. Samples that decay below
  `decay_risk_threshold` become **At-Risk** and are replayed back to the fast
  pathway (spaced repetition), which increases their review count `k` and slows
  their future decay.
* **Capacity control** — the buffer is capped at `max_buffer_size` (400). When
  full, the *least valuable* entries are evicted: those that are both strong
  (high S(t) — already consolidated) and familiar (low surprise).

### Pillar 2 — Parameter Trajectory Dynamics

* Tracks how far the **fast LoRA pathway** has drifted from its last reference
  snapshot via normalized Frobenius distance `d_θ`.
* Tracks step-to-step parameter velocity `v_t` and the **Instability Ratio**
  `r = v_t / mean(v₁..t)`.

### Pillar 3 — Instability-Gated EMA Consolidation

* When `d_θ > delta_theta_threshold`, the slow pathway absorbs a fraction of the
  fast pathway via an exponential moving average, then the reference snapshot is
  reset.
* The transfer rate `α_eff = α_base / (1 + γ·r)` is **adaptive**: when the fast
  pathway is unstable (high `r`), `α_eff` shrinks toward 0, protecting the slow
  pathway from absorbing noise.

---

## 4. Mathematical Specification

| Quantity | Formula | Symbol |
|---|---|---|
| NLL surprise (per sample) | `mean over valid j of CE(logits[b,j], labels[b,j+1])` | `S_nll` |
| Memory strength | `S(t) = exp(-λ · (t_now − t_last_seen) / k)` | `S(t)` |
| Trajectory distance | `d_θ = ‖θ_F^(t) − θ_F^(ref)‖_F / (‖θ_F^(ref)‖_F + ε)` | `d_θ` |
| Parameter velocity | `v_t = ‖θ_F^(t) − θ_F^(t−1)‖₂` | `v_t` |
| Instability ratio | `r = v_t / (mean(v₁..t) + ε)` | `r` |
| Effective EMA rate | `α_eff = α_base / (1 + γ·r) ∈ (0, α_base]` | `α_eff` |
| EMA consolidation | `θ_S ← (1 − α_eff)·θ_S + α_eff·θ_F` | — |

**Key invariants enforced in code:**

* Padding positions (`label == -100`) are *strictly excluded* from the surprise
  mean — padding can never dilute or corrupt the statistic.
* Only LoRA tensors (`lora_A` / `lora_B`) participate in dynamics and
  consolidation; all other parameters are untouched.
* `d_θ` is *normalized*, so it is scale-independent and comparable across runs.

---

## 5. Hyperparameters

Defined in `config.yaml` under `athena_hyperparams:`:

| Parameter | Value | Meaning |
|---|---|---|
| `surprise_threshold` | `2.50` | NLL above this ⇒ sample is admitted to the replay buffer |
| `decay_lambda` | `0.05` | Ebbinghaus forgetting-rate λ |
| `decay_risk_threshold` | `0.60` | S(t) below this ⇒ sample is At-Risk |
| `max_buffer_size` | `400` | Buffer capacity (60% reduction vs 1000 raw samples) |
| `delta_theta_threshold` | `0.15` | d_θ above this ⇒ consolidation triggers |
| `ema_alpha_base` | `0.20` | Maximum consolidation transfer rate α_base |
| `instability_gamma` | `2.0` | Instability sensitivity γ |

Plus global config: `tokenizer_name`, `max_seq_length` (512), `samples_per_task_train`
(1000), `samples_per_task_val` (200), `output_dir`, `seed`, `num_proc`.

Both core classes expose `from_config(config)` so they can be built directly from
`config.yaml`.

---

## 6. Data Pipeline (Step 1)

Three sequential tasks simulate domain shift:

| Task | Domain | Source datasets (first available wins) |
|---|---|---|
| `task_1_medical` | Medical | `medalpaca/medical_meadow_medqa`, `pubmed_qa` |
| `task_2_legal_finance` | Financial / Legal | `zeroshot/twitter-financial-news-sentiment`, `financial_phrasebank` |
| `task_3_superglue_boolq` | General Reasoning | `aps/super_glue` (boolq) |

Each raw row is normalized into a single templated prompt:

```
### Instruction:
<instruction>

### Context:
<context>

### Response:
<answer><|end_of_text|>
```

Then tokenized (truncate + right-pad to 512) with the Llama-3 tokenizer. The
**`labels`** are a copy of `input_ids` with every padding position replaced by
`-100`, so the loss only touches real tokens.

**Saved format** (per split): HuggingFace Arrow — `data-*.arrow`,
`dataset_info.json`, `state.json`. Features: `input_ids` (int32),
`attention_mask` (int8), `labels` (int64), all length 512.
Train = 1000 rows, Val = 200 rows per task.

These tokenized files are committed to the repo (they are small, ~6.5 MB per
train split), so a fresh clone gets the datasets immediately with no downloads.

---

## 7. Math Core (Step 2)

### `cognitive_selection_gate(logits, labels, surprise_threshold=2.5)`

Causal-LM next-token evaluation: shifts logits by one position, computes
per-token cross-entropy with `ignore_index=-100`, then averages over **valid
tokens only**. Returns `(nll_loss, high_value_mask)`.

### `CognitiveReplayBuffer`

| Method | Role |
|---|---|
| `from_config(config)` | Build from `config.yaml` |
| `add_batch(input_ids, attention_mask, labels, logits, task_id, current_step)` | Run gate, admit high-surprise samples, evict overflow; returns `(nll_loss, high_value_mask)` |
| `memory_strength(item, current_step)` | S(t) for one item |
| `get_memory_strengths(current_step)` | S(t) for the whole buffer |
| `get_at_risk_samples(current_step, batch_size)` | Rescue the weakest At-Risk samples as a collated PyTorch batch (or `None`) |
| `_evict(current_step)` | Trim to capacity by lowest value |
| `summary(current_step)` | Size / capacity / At-Risk count / mean strength |

Each buffered item stores: `input_ids`, `attention_mask`, `labels`, `task_id`,
`nll_surprise`, `last_seen_step`, `review_count`.

### `ParameterDynamicsAndConsolidation`

| Method | Role |
|---|---|
| `from_config(config)` | Build from `config.yaml` |
| `set_reference(fast_state_dict)` | Snapshot `θ_F^(ref)` |
| `track_velocity(fast_state_dict)` | Update `v_t` |
| `compute_instability_ratio()` | Compute `r` |
| `effective_alpha(r)` | Compute `α_eff` |
| `compute_trajectory_distance(fast, ref)` | Normalized `d_θ` |
| `step_consolidation(fast_state_dict, slow_state_dict)` | One step of "model time"; returns `ConsolidationResult(triggered, slow_state_dict, delta_theta, instability_ratio, alpha_eff)` |
| `summary()`, `reset_dynamics()` | Introspection / reset |

On the first call `step_consolidation` bootstraps the reference automatically.

---

## 8. Setup & Installation

Requires **Python 3.11+** (development used 3.11.9).

```bash
git clone https://github.com/AbhijithPM507/major-project.git
cd major-project

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`: `datasets`, `transformers`, `pyyaml`, `tqdm`, `torch`, `numpy`.

---

## 9. Running the Code

**Step 1 — (re)build the tokenized datasets:**

```bash
python src/data_pipeline.py
```

**Step 1 — validate the saved datasets:**

```bash
python src/verify_data.py
```

**Step 2 — run the full math verification suite:**

```bash
python src/test_step2_math.py
```

The test suite runs **CPU-only** and prints human-readable tables for each check.

---

## 10. Verification & Test Results

`python src/test_step2_math.py` → **3/3 tests passed, 0 errors**.

**Test 1 — Cognitive Gate & `-100` masking** (4 real medical samples,
logits `(4, 512, 128256)`):

* Gate reproduces designed NLL scores `[1.00, 1.80, 4.00, 6.50]` → exactly
  2 REJECTED / 2 ACCEPTED.
* **No-dilution proof**: gate NLL `0.9998` vs a naive all-position mean of
  `0.4950` — padding is genuinely excluded.
* **Poison test**: corrupting every padded logit row to `-1e4` leaves NLL and
  gate decisions bit-for-bit unchanged.
* Independent manual `log_softmax` recomputation matches
  `cross_entropy(reduction='none', ignore_index=-100)`.

**Test 2 — Decay & Replay** (λ=0.05, k=1, 50 steps):

* Threshold crossing exactly as designed: `S(10)=0.6065` healthy →
  `S(11)=0.5770` **At-Risk** → `S(50)=0.0821`.
* `get_at_risk_samples` rescued 3 items (k `1→2`, S `0.0821→1.0000`); a fresh
  sample was left untouched; returns `None` when nothing is At-Risk.
* **Spaced review** halves forgetting: at step 60, rescued `0.7788` vs
  unreviewed `0.6065`.
* **Capacity**: 450 candidates → exactly 400 kept; the lowest-NLL, highest-S
  entries were evicted first.

**Test 3 — Dynamics, Instability & EMA** (mock 8 LoRA adapters + 1 non-LoRA):

| Step | d_θ | r | α_eff | Triggered |
|---|---|---|---|---|
| bootstrap | 0.0000 | 0.0000 | 0.2000 | no |
| A: small noise | 0.0500 | 1.0000 | 0.0667 | no (slow bitwise untouched) |
| B: distribution shift | 0.4030 | 1.7778 | 0.0439 | **yes** |

* `α_eff = α_base/(1+γ·r)` verified formula-exact.
* EMA update verified elementwise; slow–fast gap shrank by exactly
  `(1 − α_eff) = 0.9561`; non-LoRA tensors untouched.
* Reference reset to `d_θ = 0`; normalization confirmed by `d(2·ref, ref)=1.0`.

---

## 11. Known Environment Notes

* **Import order matters on Windows.** Importing `torch` *before*
  `datasets`/`pyarrow` can crash the interpreter with an access violation
  (`0xC0000005`). Always import `datasets` first, then `torch` — the test suite
  already does this.
* `num_proc: 1` in `config.yaml` avoids multiprocessing tokenization issues;
  the pipeline falls back to single-process automatically if needed.
* The data pipeline uses `trust_remote_code=False` and quotes the tokenizer's
  EOS token as the pad token.

---

## 12. Roadmap

| Step | Status | Description |
|---|---|---|
| Step 1 | ✅ Done | Tokenized sequential datasets |
| Step 2 | ✅ Done | Cognitive gate, replay buffer, trajectory dynamics, EMA consolidation |
| Step 3+ | ⏳ Planned | Wire the math core onto a real dual-LoRA Llama-3 model, run the sequential training loop, and evaluate forgetting/retention on the three task domains |

---

## License & Attribution

Academic major-project work. Third-party datasets and the
`NousResearch/Meta-Llama-3-8B-Instruct` tokenizer retain their original licenses.
