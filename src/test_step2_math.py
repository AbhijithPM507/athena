"""ATHENA Step 2 verification suite.

CPU-only unit tests for the three research pillars:

  Test 1 - cognitive_selection_gate(): NLL surprise scoring on REAL tokenized
           medical samples with strict -100 padding masking.
  Test 2 - CognitiveReplayBuffer: Ebbinghaus memory-strength decay over 50
           simulated steps, at-risk rescue, and capacity (60% reduction) enforcement.
  Test 3 - ParameterDynamicsAndConsolidation: trajectory distance, instability
           ratio r, adaptive alpha_eff and dynamics-triggered EMA consolidation.

Run:  python src/test_step2_math.py
"""

from __future__ import annotations

import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from datasets import load_from_disk  # must precede torch on this machine (Windows DLL conflict)
from transformers import AutoTokenizer
import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cognitive_gate import CognitiveReplayBuffer, cognitive_selection_gate
from dynamics_and_ema import (
    ParameterDynamicsAndConsolidation,
    compute_trajectory_distance,
    select_lora_tensors,
)


def header(title: str) -> None:
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


def passed(message: str) -> None:
    print(f"  [PASS] {message}")


def design_logits_for_targets(
    labels: torch.Tensor, target_nlls: list[float], vocab_size: int
) -> torch.Tensor:
    """Logits (B, T, V) whose per-sample mean NLL over valid tokens equals target_nlls exactly.

    At every valid (non -100) shifted position the true-class logit is set to

        a = ln(V - 1) - ln(exp(target) - 1)

    and every other class logit to 0, which yields

        NLL = -log( exp(a) / (exp(a) + V - 1) ) = target.
    """
    batch_size, seq_len = labels.shape
    logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32)
    shift_labels = labels[..., 1:]
    valid = shift_labels != -100
    bonus = torch.tensor(
        [
            math.log(vocab_size - 1) - math.log(math.expm1(float(target)))
            for target in target_nlls
        ],
        dtype=torch.float32,
    )
    positions = valid.nonzero(as_tuple=False)
    rows, cols = positions[:, 0], positions[:, 1]
    classes = shift_labels[valid]
    logits[rows, cols, classes] = bonus[rows]
    return logits


def synthetic_batch(
    num: int,
    seq_len: int,
    valid_len: int,
    target_nlls: list[float],
    vocab_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, vocab_size, (num, seq_len), generator=generator)
    attention_mask = torch.zeros(num, seq_len, dtype=torch.int8)
    attention_mask[:, :valid_len] = 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    logits = design_logits_for_targets(labels, target_nlls, vocab_size)
    return input_ids, attention_mask, labels, logits


# =========================================================================== #
# TEST 1: Cognitive Gate & -100 Masking                                       #
# =========================================================================== #
def test_1_cognitive_gate(config: dict) -> None:
    header("TEST 1 - Cognitive Gate & -100 Masking (batch of 4 REAL tokenized samples)")
    threshold = float(config["athena_hyperparams"]["surprise_threshold"])

    dataset_path = PROJECT_ROOT / "data" / "processed" / "task_1_medical" / "train"
    dataset = load_from_disk(str(dataset_path))
    dataset.set_format("torch")
    samples = dataset[:4]
    input_ids = samples["input_ids"]
    attention_mask = samples["attention_mask"]
    labels = samples["labels"]
    assert tuple(input_ids.shape) == (4, 512), f"expected (4, 512), got {tuple(input_ids.shape)}"
    assert labels.dtype == torch.int64, f"labels must be int64, got {labels.dtype}"
    passed(f"Loaded 4 real samples from {dataset_path.as_posix()} - tensors shape (4, 512), CPU only")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
        vocab_size = len(tokenizer)
    except Exception as error:
        vocab_size = int(labels.max().item()) + 1
        print(f"  [WARN] tokenizer unavailable ({type(error).__name__}); using vocab_size={vocab_size}")
    assert int(labels.max().item()) < vocab_size, "label id outside vocabulary"
    passed(f"Vocabulary size = {vocab_size}")

    assert torch.equal(labels.eq(-100), attention_mask.eq(0)), "labels==-100 must coincide with attention_mask==0"
    padded_total = int(attention_mask.eq(0).sum())
    assert padded_total > 0, "batch unexpectedly contains no padding"
    passed(f"Step-1 invariant re-checked: labels==-100 EXACTLY where attention_mask==0 ({padded_total} padded positions)")

    target_nlls = [1.00, 1.80, 4.00, 6.50]
    logits = design_logits_for_targets(labels, target_nlls, vocab_size)
    print(f"  Simulated logits: shape {tuple(logits.shape)}, {logits.numel() * 4 / 1024 ** 3:.2f} GiB float32, "
          f"designed per-sample NLL = {target_nlls}")

    with torch.no_grad():
        nll_loss, high_value_mask = cognitive_selection_gate(logits, labels, threshold)

    assert tuple(nll_loss.shape) == (4,) and tuple(high_value_mask.shape) == (4,)
    assert high_value_mask.dtype == torch.bool
    assert torch.allclose(nll_loss, torch.tensor(target_nlls), atol=1e-2), (
        f"designed NLL scores not reproduced: {nll_loss.tolist()}"
    )
    expected_mask = [False, False, True, True]
    assert high_value_mask.tolist() == expected_mask, (
        f"gate decisions {high_value_mask.tolist()} != {expected_mask}"
    )
    passed("Gate reproduces the 4 designed NLL scores (1.00 / 1.80 / 4.00 / 6.50) and yields exactly 2 rejected / 2 accepted")

    shift_labels = labels[..., 1:]
    valid_counts = shift_labels.ne(-100).sum(dim=1)
    for b in range(4):
        shift_log_probs = torch.log_softmax(logits[b, :-1].float(), dim=-1)
        valid_rows = shift_labels[b].ne(-100)
        per_token_nll = -shift_log_probs[valid_rows, shift_labels[b][valid_rows]]
        manual_nll = float(per_token_nll.mean())
        assert abs(manual_nll - float(nll_loss[b])) < 1e-3, (
            f"manual log-softmax NLL {manual_nll} != gate NLL {float(nll_loss[b])} on sample {b}"
        )
        assert 0 < int(valid_counts[b]) < 512, "sample must contain both valid and padded tokens"
        del shift_log_probs, per_token_nll
    passed("Independent manual log-softmax recomputation matches cross_entropy(reduction='none', ignore_index=-100) mean for all 4 samples")

    diluted = nll_loss * valid_counts.float() / 511.0
    assert torch.all(diluted < nll_loss), "padding would have diluted the NLL if not excluded"
    print("  No-dilution evidence (mean NLL over VALID tokens only vs naive mean over all 511 shift positions):")
    for b in range(4):
        print(f"    sample {b}: gate NLL = {float(nll_loss[b]):9.4f}   naive all-position mean = {float(diluted[b]):9.4f}")

    pad_rows = (shift_labels == -100).nonzero(as_tuple=False)
    logits[pad_rows[:, 0], pad_rows[:, 1], :] = -1e4
    with torch.no_grad():
        nll_poisoned, mask_poisoned = cognitive_selection_gate(logits, labels, threshold)
    assert torch.allclose(nll_poisoned, nll_loss, atol=1e-5), "padding positions leaked into the NLL!"
    assert torch.equal(mask_poisoned, high_value_mask), "gate decision changed after padding poison!"
    passed("Poison test: corrupting every padded logit row (-1e4) leaves NLL and gate decisions unchanged - padding cannot touch the surprise statistic")

    print()
    print("    +--------+----------+-------------+---------------+-----------+------------------------------------------+")
    print("    | Sample | Padded   | Valid Toks  | NLL Surprise  | Threshold | Gate Decision                            |")
    print("    +--------+----------+-------------+---------------+-----------+------------------------------------------+")
    for b in range(4):
        decision = (
            "ACCEPTED (High Surprise - At Risk)"
            if high_value_mask[b]
            else "REJECTED (Familiar / Low Surprise)"
        )
        print(
            f"    | {b:<6} | {int(attention_mask[b].eq(0).sum()):<8} | "
            f"{int(valid_counts[b]):<11} | {float(nll_loss[b]):>13.4f} | {threshold:>9.2f} | {decision:<40} |"
        )
    print("    +--------+----------+-------------+---------------+-----------+------------------------------------------+")
    del logits, nll_poisoned


# =========================================================================== #
# TEST 2: Memory Strength Decay & Replay                                      #
# =========================================================================== #
def test_2_decay_and_replay(config: dict) -> None:
    header("TEST 2 - Memory Strength Decay & Replay Buffer Rescue (50 simulated steps)")
    hyper = config["athena_hyperparams"]
    buffer = CognitiveReplayBuffer.from_config(config)
    seq_len, valid_len, small_vocab = 16, 10, 64

    print("  Criterion 1 - Novelty via Surprise (admission gate):")
    targets = [0.50, 1.20, 3.00, 4.50, 2.00, 5.50]
    ids, mask, labels, logits = synthetic_batch(6, seq_len, valid_len, targets, small_vocab, seed=7)
    nll, accepted = buffer.add_batch(ids, mask, labels, logits, task_id=1, current_step=0)
    assert accepted.tolist() == [False, False, True, True, False, True], (
        f"admission decisions wrong: {accepted.tolist()}"
    )
    assert len(buffer) == 3
    stored_nlls = sorted(item["nll_surprise"] for item in buffer.buffer)
    assert all(abs(a - b) < 1e-2 for a, b in zip(stored_nlls, [3.0, 4.5, 5.5]))
    print("      +--------+---------------+-----------+------------------------------------------+")
    print("      | Sample | Designed NLL  | Threshold | Buffer Decision                           |")
    print("      +--------+---------------+-----------+------------------------------------------+")
    for b in range(6):
        decision = (
            "ACCEPTED (High Surprise - At Risk)"
            if accepted[b]
            else "REJECTED (Familiar / Low Surprise)"
        )
        print(f"      | {b:<6} | {targets[b]:>13.4f} | {2.5:>9.2f} | {decision:<40} |")
    print("      +--------+---------------+-----------+------------------------------------------+")
    passed(f"Criterion 1 admitted exactly the high-surprise samples ({len(buffer)}/6 stored)")

    print()
    print(f"  Criterion 2 - Ebbinghaus decay S(t) = exp(-{hyper['decay_lambda']} * dt / k), "
          f"items added at step 0 with k=1 (risk threshold {hyper['decay_risk_threshold']}):")
    print("      +------+-----------+-----------+------------------+")
    print("      | Step |   S(t)    | At-Risk?  | Status           |")
    print("      +------+-----------+-----------+------------------+")
    for step in (0, 5, 10, 11, 25, 50):
        strengths = buffer.get_memory_strengths(step)
        assert len(strengths) == 3
        strength = strengths[0]
        at_risk = strength < hyper["decay_risk_threshold"]
        status = (
            "AT-RISK" if at_risk else ("Fresh" if step == 0 else "Healthy")
        )
        print(f"      | {step:>4} | {strength:9.4f} | {'YES' if at_risk else 'no':>9}  | {status:<16} |")
    print("      +------+-----------+-----------+------------------+")
    s10 = buffer.get_memory_strengths(10)
    s11 = buffer.get_memory_strengths(11)
    s50 = buffer.get_memory_strengths(50)
    assert all(s >= 0.60 for s in s10), "S(10) should still be above the risk threshold"
    assert all(s < 0.60 for s in s11), "S(11) should cross below the risk threshold"
    assert all(s < 0.60 for s in s50), "S(50) should be deep below the risk threshold"
    passed("Threshold crossing observed: S(10)=0.6065 (healthy) -> S(11)=0.5770 (At-Risk) -> S(50)=0.0821")

    assert buffer.get_at_risk_samples(current_step=4, batch_size=10) is None, (
        "nothing is At-Risk at step 4 - method must return None"
    )
    passed("get_at_risk_samples(step=4) returns None - healthy memories are not rescued prematurely")

    ids_fresh, mask_fresh, labels_fresh, logits_fresh = synthetic_batch(
        1, seq_len, valid_len, [3.2], small_vocab, seed=8
    )
    buffer.add_batch(ids_fresh, mask_fresh, labels_fresh, logits_fresh, task_id=2, current_step=50)
    assert len(buffer) == 4
    passed("A fresh high-surprise sample (S=1.0) was added at step 50 - it must NOT be rescued")

    rescue_batch = buffer.get_at_risk_samples(current_step=50, batch_size=10)
    assert rescue_batch is not None, "At-Risk samples exist at step 50 - rescue batch must not be None"
    rescued_indices = rescue_batch["buffer_indices"]
    assert rescued_indices == [0, 1, 2], f"rescued {rescued_indices}, expected the 3 decaying originals"
    assert tuple(rescue_batch["input_ids"].shape) == (3, seq_len)
    assert tuple(rescue_batch["attention_mask"].shape) == (3, seq_len)
    assert tuple(rescue_batch["labels"].shape) == (3, seq_len)
    assert rescue_batch["input_ids"].dtype == torch.int64 and rescue_batch["labels"].dtype == torch.int64
    assert int((rescue_batch["review_count"] == 2).sum()) == 3, "review_count k must be incremented 1 -> 2"
    assert int((rescue_batch["last_seen_step"] == 50).sum()) == 3, "last_seen_step must equal current_step"
    expected_before = math.exp(-hyper["decay_lambda"] * 50 / 1)
    assert torch.allclose(
        rescue_batch["memory_strength_before"], torch.full((3,), expected_before), atol=1e-6
    ), "rescued items must have been below the risk threshold before the update"
    strengths_after = buffer.get_memory_strengths(50)
    for index in rescued_indices:
        assert strengths_after[index] == 1.0, (
            f"memory strength of rescued item {index} must reset to 1.0, got {strengths_after[index]}"
        )
    assert buffer.buffer[3]["review_count"] == 1, "the fresh item must remain untouched"
    print()
    print("  Rescue operation get_at_risk_samples(current_step=50, batch_size=10):")
    print("      +---------+---------+-------------+---------------------+------------------------+")
    print("      | Item    | task_id | NLL         | k (before -> after) | S(t) (before -> after) |")
    print("      +---------+---------+-------------+---------------------+------------------------+")
    for position, index in enumerate(rescued_indices):
        item = buffer.buffer[index]
        print(
            f"      | #{index:<6} | {item['task_id']:>7} | {item['nll_surprise']:>11.4f} | "
            f"{1:>19} -> {item['review_count']} | {float(rescue_batch['memory_strength_before'][position]):>12.4f} -> 1.0000 |"
        )
    print("      +---------+---------+-------------+---------------------+------------------------+")
    passed("3 At-Risk items rescued as a (3, 16) PyTorch batch for the Fast Pathway: k 1 -> 2, "
           "last_seen 0 -> 50, S(t) reset 0.0821 -> 1.0000; fresh item untouched")

    s60 = buffer.get_memory_strengths(60)
    rescued_s60 = math.exp(-hyper["decay_lambda"] * 10 / 2)
    fresh_s60 = math.exp(-hyper["decay_lambda"] * 10 / 1)
    assert all(abs(s60[index] - rescued_s60) < 1e-9 for index in rescued_indices)
    assert abs(s60[3] - fresh_s60) < 1e-9
    assert buffer.get_at_risk_samples(current_step=60, batch_size=10) is None
    print()
    print("  Review effect at step 60 (10 steps after the rescue):")
    print(f"    rescued items (k=2): S = exp(-0.05*10/2) = {rescued_s60:.4f}")
    print(f"    unreviewed item (k=1): S = exp(-0.05*10/1) = {fresh_s60:.4f}   <- reviewing SLOWS forgetting")
    passed("spaced review halves the decay rate; nothing is At-Risk at step 60 -> get_at_risk_samples returns None")

    cap_buffer = CognitiveReplayBuffer.from_config(config)
    all_nlls: list[float] = []
    chunk_size = 50
    for chunk in range(9):
        designed = [2.6 + 0.002 * (chunk * chunk_size + i) for i in range(chunk_size)]
        ids, mask, labels, logits = synthetic_batch(
            chunk_size, seq_len, valid_len, designed, small_vocab, seed=100 + chunk
        )
        nll, accepted = cap_buffer.add_batch(ids, mask, labels, logits, task_id=3, current_step=0)
        assert bool(accepted.all()), "every designed sample must exceed the surprise threshold"
        all_nlls.extend(nll.tolist())
        assert len(cap_buffer) <= cap_buffer.max_buffer_size, "buffer bloated beyond capacity!"
    assert len(cap_buffer) == cap_buffer.max_buffer_size == 400
    survivors = [item["nll_surprise"] for item in cap_buffer.buffer]
    evicted_counter = Counter(round(v, 5) for v in all_nlls) - Counter(round(v, 5) for v in survivors)
    assert sum(evicted_counter.values()) == 50, f"expected 50 evictions, saw {sum(evicted_counter.values())}"
    assert min(survivors) >= max(evicted_counter), (
        "eviction must remove the LOWEST-NLL items first (equal S(t) at admission)"
    )
    print()
    print(f"  Capacity optimization (60% buffer reduction target: 400 vs 1000 raw samples):")
    print(f"    450 high-surprise samples offered -> {len(cap_buffer)} kept, "
          f"{sum(evicted_counter.values())} evicted (least valuable = highest S(t) + lowest NLL)")
    passed("Capacity never exceeded: buffer holds exactly 400/450; the 50 lowest-NLL items were evicted first")

    mini = CognitiveReplayBuffer(
        surprise_threshold=2.5,
        decay_lambda=hyper["decay_lambda"],
        decay_risk_threshold=hyper["decay_risk_threshold"],
        max_buffer_size=2,
    )
    ids, mask, labels, logits = synthetic_batch(1, seq_len, valid_len, [5.0], small_vocab, seed=1)
    mini.add_batch(ids, mask, labels, logits, task_id=9, current_step=0)  # item A -> will decay
    ids, mask, labels, logits = synthetic_batch(2, seq_len, valid_len, [5.0, 5.0], small_vocab, seed=2)
    mini.add_batch(ids, mask, labels, logits, task_id=9, current_step=50)  # items B, C -> fresh, overflow
    assert len(mini) == 2
    mini_strengths = mini.get_memory_strengths(50)
    assert any(s < hyper["decay_risk_threshold"] for s in mini_strengths), (
        "the stale At-Risk item must survive eviction (its weak trace is precious)"
    )
    assert sum(1 for s in mini_strengths if s > 0.99) == 1, "exactly one strong fresh item must remain"
    passed("Eviction preference verified: kept the WEAK stale item S=0.0821 (At-Risk, precious) "
           "and evicted a STRONG fresh item S=1.0000 (highest S(t) = least valuable)")


# =========================================================================== #
# TEST 3: Dynamics d_theta & EMA Consolidation                                #
# =========================================================================== #
def make_mock_lora_state(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {}
    for layer in range(2):
        for projection in ("q_proj", "v_proj"):
            base = f"base_model.model.layers.{layer}.self_attn.{projection}"
            state[f"{base}.lora_A.weight"] = torch.randn(8, 512, generator=generator) * 0.02
            state[f"{base}.lora_B.weight"] = torch.randn(512, 8, generator=generator) * 0.02
    state["modules_to_save.default.score.weight"] = torch.randn(4, 512, generator=generator)
    return state


def lora_frobenius_norm(state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in select_lora_tensors(state).values():
        total += float((tensor.detach().float() ** 2).sum())
    return math.sqrt(total)


def scaled_random_lora_noise(
    state: dict[str, torch.Tensor], generator: torch.Generator, target_norm: float
) -> dict[str, torch.Tensor]:
    noise: dict[str, torch.Tensor] = {}
    total = 0.0
    for key in select_lora_tensors(state):
        tensor = torch.randn(state[key].shape, generator=generator)
        noise[key] = tensor
        total += float((tensor**2).sum())
    scale = target_norm / math.sqrt(total)
    return {key: value * scale for key, value in noise.items()}


def apply_lora_noise(
    state: dict[str, torch.Tensor], noise: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        key: (value + noise[key] if key in noise else value)
        for key, value in state.items()
    }


def test_3_dynamics_and_ema(config: dict) -> None:
    header("TEST 3 - Trajectory d_theta, Instability Ratio r & EMA Consolidation")
    hyper = config["athena_hyperparams"]
    torch.manual_seed(42)

    fast_state = make_mock_lora_state(seed=0)
    slow_state = make_mock_lora_state(seed=1)
    lora = select_lora_tensors(fast_state)
    assert tuple(lora["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"].shape) == (8, 512)
    assert tuple(lora["base_model.model.layers.0.self_attn.q_proj.lora_B.weight"].shape) == (512, 8)
    assert len(lora) == 8
    non_lora_key = "modules_to_save.default.score.weight"
    assert non_lora_key in fast_state and non_lora_key not in lora
    ref_norm = lora_frobenius_norm(fast_state)
    print(f"  Mock LoRA pathways: fast/slow each hold 8 adapter tensors "
          f"(4x lora_A (8, 512) + 4x lora_B (512, 8)) + 1 non-LoRA tensor")
    print(f"  ||theta_F^(ref)||_F = {ref_norm:.4f}")

    d_identical = compute_trajectory_distance(fast_state, fast_state)
    assert d_identical == 0.0
    scaled = {key: value * 2 for key, value in fast_state.items()}
    d_scaled = compute_trajectory_distance(scaled, fast_state)
    assert abs(d_scaled - 1.0) < 1e-4, f"d(2*ref, ref) should be ~1, got {d_scaled}"
    perturbed_non_lora = {
        key: (value + 100.0 if key == non_lora_key else value)
        for key, value in fast_state.items()
    }
    d_non_lora = compute_trajectory_distance(perturbed_non_lora, fast_state)
    assert d_non_lora == 0.0, "non-LoRA tensors must be excluded from d_theta"
    passed(f"compute_trajectory_distance: d(fast,fast)={d_identical:.6f}, d(2*fast,fast)={d_scaled:.6f} "
           f"(normalized), non-LoRA keys excluded (d={d_non_lora:.6f})")

    dynamics = ParameterDynamicsAndConsolidation.from_config(config)
    assert dynamics.delta_theta_threshold == 0.15
    assert dynamics.ema_alpha_base == 0.20
    assert dynamics.instability_gamma == 2.0
    passed("Hyperparameters loaded: delta_theta_threshold=0.15, ema_alpha_base=0.20, instability_gamma=2.0")

    result_bootstrap = dynamics.step_consolidation(fast_state, slow_state)
    assert result_bootstrap.triggered is False
    assert result_bootstrap.delta_theta == 0.0
    assert result_bootstrap.instability_ratio == 0.0
    assert len(dynamics.velocities) == 0, "bootstrap step has no t-1 snapshot -> no velocity yet"
    passed("Bootstrap: reference snapshot theta_F^(ref) established (d_theta = 0, no consolidation)")

    slow_before = {key: value.clone() for key, value in slow_state.items()}
    generator = torch.Generator().manual_seed(1234)

    small_noise = scaled_random_lora_noise(fast_state, generator, 0.05 * ref_norm)
    fast_small = apply_lora_noise(fast_state, small_noise)
    result_a = dynamics.step_consolidation(fast_small, slow_state)
    assert result_a.triggered is False, "small perturbation must NOT trigger consolidation"
    assert 0.0 < result_a.delta_theta < 0.15, (
        f"d_theta for step A must lie below the 0.15 threshold, got {result_a.delta_theta}"
    )
    assert 0.0 < result_a.alpha_eff <= 0.20
    assert result_a.slow_state_dict is slow_state
    assert all(torch.equal(slow_state[key], slow_before[key]) for key in slow_state), (
        "slow_state_dict must stay bitwise untouched when not triggered"
    )
    passed(f"Step A: d_theta={result_a.delta_theta:.4f} < 0.15 -> triggered=False, "
           f"slow_state_dict bitwise untouched (r={result_a.instability_ratio:.4f}, "
           f"alpha_eff={result_a.alpha_eff:.4f})")

    shift_noise = scaled_random_lora_noise(fast_state, generator, 0.40 * ref_norm)
    fast_shifted = apply_lora_noise(fast_small, shift_noise)
    result_b = dynamics.step_consolidation(fast_shifted, slow_state)
    assert result_b.triggered is True, "distributional shift must trigger consolidation"
    assert result_b.delta_theta > 0.15, (
        f"d_theta for step B must exceed the 0.15 threshold, got {result_b.delta_theta}"
    )
    assert result_b.instability_ratio > result_a.instability_ratio, (
        "velocity spike must raise the instability ratio"
    )
    expected_alpha = 0.20 / (1.0 + 2.0 * result_b.instability_ratio)
    assert abs(result_b.alpha_eff - expected_alpha) < 1e-9, "alpha_eff must equal alpha_base/(1+gamma*r)"
    assert 0.0 < result_b.alpha_eff < result_a.alpha_eff, (
        "higher instability must SHRINK the EMA transfer rate (Slow pathway shielding)"
    )

    score_key_before = slow_before[non_lora_key].clone()
    for key, fast_tensor in select_lora_tensors(fast_shifted).items():
        expected = (1.0 - result_b.alpha_eff) * slow_before[key] + result_b.alpha_eff * fast_tensor
        assert torch.allclose(slow_state[key], expected, atol=1e-6), f"EMA mismatch on {key}"
    assert torch.equal(slow_state[non_lora_key], score_key_before), "non-LoRA tensors must not be updated"

    d_slow_before = compute_trajectory_distance(slow_before, fast_shifted)
    d_slow_after = compute_trajectory_distance(slow_state, fast_shifted)
    assert d_slow_after < d_slow_before, "slow pathway must move TOWARD the fast pathway"
    expected_ratio = 1.0 - result_b.alpha_eff
    assert abs(d_slow_after - expected_ratio * d_slow_before) < 1e-5, (
        "EMA must shrink the slow-fast gap by exactly (1 - alpha_eff)"
    )

    d_after_reset = compute_trajectory_distance(fast_shifted, dynamics.reference_state_dict)
    assert d_after_reset <= 1e-9, "reference snapshot must be reset to the current fast pathway"
    assert dynamics.consolidation_count == 1

    print()
    print("    +---------------------+-----------+------------------+--------------+-----------+")
    print("    | Consolidation step  |  d_theta  | Instability r    |  alpha_eff   | Triggered |")
    print("    +---------------------+-----------+------------------+--------------+-----------+")
    print(f"    | bootstrap           | {result_bootstrap.delta_theta:>9.4f}  | {result_bootstrap.instability_ratio:>16.4f}  | "
          f"{result_bootstrap.alpha_eff:>12.4f}  | no        |")
    print(f"    | A: small noise      | {result_a.delta_theta:>9.4f}  | {result_a.instability_ratio:>16.4f}  | "
          f"{result_a.alpha_eff:>12.4f}  | no        |")
    print(f"    | B: distributional   | {result_b.delta_theta:>9.4f}  | {result_b.instability_ratio:>16.4f}  | "
          f"{result_b.alpha_eff:>12.4f}  | YES       |")
    print("    +---------------------+-----------+------------------+--------------+-----------+")
    passed(f"Step B: d_theta={result_b.delta_theta:.4f} > 0.15 -> triggered=True, r={result_b.instability_ratio:.4f}, "
           f"alpha_eff={result_b.alpha_eff:.4f} ({expected_alpha / result_b.alpha_eff:.2f}x formula-exact)")
    passed("EMA verified elementwise: theta_S <- (1-alpha_eff)*theta_S + alpha_eff*theta_F; "
           f"slow-fast gap shrank by exactly (1-alpha_eff)={expected_ratio:.4f}; non-LoRA tensors untouched")
    passed(f"Reference reset: d_theta(fast_shifted, theta_F^(ref)) = {d_after_reset:.10f} -> 0")

    result_c = dynamics.step_consolidation(fast_shifted, slow_state)
    assert result_c.triggered is False
    assert result_c.delta_theta <= 1e-9
    assert result_c.instability_ratio == 0.0
    passed("Post-consolidation stability: identical fast weights -> d_theta=0, v_t=0, r=0, no re-trigger")

    print()
    print("    Parameter velocity history v_t = ||theta_F^(t) - theta_F^(t-1)||_2:")
    for index, velocity in enumerate(dynamics.velocities, start=1):
        print(f"      step {index}: {velocity:.6f}")
    print()
    print("    Adaptive EMA mapping  alpha_eff = 0.20 / (1 + 2.0 * r):")
    for ratio in (0.0, 0.5, 1.0, round(result_b.instability_ratio, 4), 5.0, 10.0):
        print(f"      r = {ratio:8.4f}  ->  alpha_eff = {dynamics.effective_alpha(ratio):.4f}")
    alphas = [dynamics.effective_alpha(ratio) for ratio in (0.0, 0.5, 1.0, 5.0, 10.0)]
    assert alphas[0] == 0.20
    assert all(alphas[i] > alphas[i + 1] for i in range(len(alphas) - 1)), "alpha_eff must be strictly decreasing in r"
    assert all(0.0 < alpha <= 0.20 for alpha in alphas)
    passed("Design check: chaotic fast pathway (high r) -> alpha_eff shrinks toward 0, shielding the Slow pathway; "
           "stable pathway (low r) -> alpha_eff approaches alpha_base")


def main() -> int:
    started = time.time()
    print("=" * 100)
    print("  ATHENA STEP 2 VERIFICATION SUITE - Core Math Engines, Selection Gate & Consolidation (CPU-only)")
    print("=" * 100)
    with (PROJECT_ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    hyper = config["athena_hyperparams"]
    print("  athena_hyperparams loaded from config.yaml:")
    for key, value in hyper.items():
        print(f"    {key:<24} = {value}")
    print(f"  torch {torch.__version__} | device = cpu | datasets {__import__('datasets').__version__}")

    tests = (
        ("Test 1 - Cognitive Gate & -100 Masking", test_1_cognitive_gate),
        ("Test 2 - Memory Strength Decay & Replay Rescue", test_2_decay_and_replay),
        ("Test 3 - Dynamics, Instability Ratio & EMA Consolidation", test_3_dynamics_and_ema),
    )
    results: list[tuple[str, bool]] = []
    for name, test in tests:
        try:
            test(config)
            results.append((name, True))
        except Exception as error:
            results.append((name, False))
            print(f"\n  [ERROR] {name} failed: {error}")
            traceback.print_exc()

    print()
    print("=" * 100)
    print("  STEP 2 VERIFICATION SUMMARY")
    print("=" * 100)
    for name, ok in results:
        dots = "." * max(3, 70 - len(name))
        print(f"   {name} {dots} {'PASS' if ok else 'FAIL'}")
    errors = sum(1 for _, ok in results if not ok)
    print("=" * 100)
    if errors == 0:
        print(f"   TOTAL: {len(results)}/{len(results)} tests passed - 0 errors  ({time.time() - started:.2f}s)")
    else:
        print(f"   TOTAL: {len(results) - errors}/{len(results)} tests passed - {errors} ERROR(S)")
    print("=" * 100)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
