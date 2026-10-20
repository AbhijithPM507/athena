"""Pillar 1 - Cognitive-Aware Selection Gate and Consolidation Replay Buffer.

Modules
-------
cognitive_selection_gate(logits, labels, surprise_threshold=2.5)
    Per-sample Negative Log-Likelihood (NLL) surprise scoring for causal-LM
    next-token evaluation. Padding positions carry the label ``-100`` and are
    strictly excluded from the mean so padding can never corrupt the
    Negative Log-Likelihood Surprise statistic.

CognitiveReplayBuffer
    Dual-criteria consolidation buffer:
      Criterion 1 - Novelty: only samples whose NLL surprise exceeds
                     ``surprise_threshold`` are admitted.
      Criterion 2 - Decay:   stored samples are tracked with the biological
                     Ebbinghaus retention formula

                         S(t) = exp(-lambda * (t_current - t_last_seen) / k)

                     where ``k`` is the review count. Samples whose memory
                     strength falls below ``decay_risk_threshold`` are
                     flagged At-Risk and rescued by ``get_at_risk_samples``.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def cognitive_selection_gate(logits, labels, surprise_threshold=2.5):
    # Shift logits and labels for causal LM next-token evaluation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Calculate per-token Negative Log-Likelihood (NLL) ignoring -100 padding
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size(0), -1)

    # Compute mean NLL per sample ONLY over valid (non -100) tokens
    valid_mask = (shift_labels != -100).float()
    valid_token_counts = valid_mask.sum(dim=1).clamp(min=1.0)
    nll_loss = (token_losses * valid_mask).sum(dim=1) / valid_token_counts

    # Filter based on the surprise threshold
    high_value_mask = nll_loss > surprise_threshold
    return nll_loss, high_value_mask


class CognitiveReplayBuffer:
    """Consolidation Replay Buffer with surprise admission and Ebbinghaus decay."""

    def __init__(
        self,
        surprise_threshold: float = 2.5,
        decay_lambda: float = 0.05,
        decay_risk_threshold: float = 0.60,
        max_buffer_size: int = 400,
    ) -> None:
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be >= 1")
        if decay_lambda <= 0.0:
            raise ValueError("decay_lambda must be > 0")
        self.surprise_threshold = float(surprise_threshold)
        self.decay_lambda = float(decay_lambda)
        self.decay_risk_threshold = float(decay_risk_threshold)
        self.max_buffer_size = int(max_buffer_size)
        self.buffer: list[dict[str, Any]] = []
        self.stats: dict[str, int] = {
            "accepted": 0,
            "rejected": 0,
            "evicted": 0,
            "rescued": 0,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> "CognitiveReplayBuffer":
        params = config.get("athena_hyperparams", config) if isinstance(config, dict) else {}
        allowed = {
            "surprise_threshold",
            "decay_lambda",
            "decay_risk_threshold",
            "max_buffer_size",
        }
        kwargs = {key: value for key, value in dict(params).items() if key in allowed}
        kwargs.update(overrides)
        return cls(**kwargs)

    def __len__(self) -> int:
        return len(self.buffer)

    # ------------------------------------------------------------------ #
    # Criterion 1: Novelty via Surprise                                   #
    # ------------------------------------------------------------------ #
    def add_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        task_id: int,
        current_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate an incoming batch with the gate and admit high-surprise samples.

        Returns
        -------
        (nll_loss, high_value_mask) as produced by :func:`cognitive_selection_gate`.
        """
        nll_loss, high_value_mask = cognitive_selection_gate(
            logits, labels, self.surprise_threshold
        )
        for index in torch.nonzero(high_value_mask, as_tuple=False).flatten().tolist():
            self.buffer.append(
                {
                    "input_ids": torch.as_tensor(input_ids[index]).detach().cpu().clone(),
                    "attention_mask": torch.as_tensor(attention_mask[index]).detach().cpu().clone(),
                    "labels": torch.as_tensor(labels[index]).detach().cpu().clone(),
                    "task_id": int(task_id),
                    "nll_surprise": float(nll_loss[index].item()),
                    "last_seen_step": int(current_step),
                    "review_count": 1,
                }
            )
            self.stats["accepted"] += 1
        self.stats["rejected"] += int((~high_value_mask).sum().item())
        self._evict(current_step)
        return nll_loss, high_value_mask

    # ------------------------------------------------------------------ #
    # Criterion 2: Memory Strength Decay (Ebbinghaus retention)           #
    # ------------------------------------------------------------------ #
    def memory_strength(self, item: dict[str, Any], current_step: int) -> float:
        """S(t) = exp(-lambda * (t_current - t_last_seen) / k)."""
        elapsed = max(0, int(current_step) - int(item["last_seen_step"]))
        review_count = max(1, int(item["review_count"]))
        return math.exp(-self.decay_lambda * elapsed / review_count)

    def get_memory_strengths(self, current_step: int) -> list[float]:
        return [self.memory_strength(item, current_step) for item in self.buffer]

    def get_at_risk_samples(
        self, current_step: int, batch_size: int
    ) -> dict[str, Any] | None:
        """Rescue the weakest (lowest S(t)) samples that fell below the risk threshold.

        For every selected item:
          * review_count  k <- k + 1   (reviewing strengthens the trace)
          * last_seen_step <- current_step  (S(t) resets to exactly 1.0)

        Returns the selected items collated as a PyTorch batch ready to be fed
        back into the Fast (Plastic) Pathway, or ``None`` when no sample is
        currently At-Risk.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        scored = sorted(
            (
                (self.memory_strength(item, current_step), index)
                for index, item in enumerate(self.buffer)
            ),
            key=lambda pair: (pair[0], pair[1]),
        )
        at_risk = [pair for pair in scored if pair[0] < self.decay_risk_threshold]
        selected = at_risk[:batch_size]
        if not selected:
            return None

        items: list[dict[str, Any]] = []
        indices: list[int] = []
        strengths_before: list[float] = []
        for strength, index in selected:
            item = self.buffer[index]
            strengths_before.append(strength)
            item["review_count"] += 1
            item["last_seen_step"] = int(current_step)
            items.append(item)
            indices.append(index)
            self.stats["rescued"] += 1

        return {
            "input_ids": torch.stack([item["input_ids"] for item in items]),
            "attention_mask": torch.stack([item["attention_mask"] for item in items]),
            "labels": torch.stack([item["labels"] for item in items]),
            "task_id": [item["task_id"] for item in items],
            "nll_surprise": torch.tensor(
                [item["nll_surprise"] for item in items], dtype=torch.float32
            ),
            "review_count": torch.tensor(
                [item["review_count"] for item in items], dtype=torch.long
            ),
            "last_seen_step": torch.tensor(
                [item["last_seen_step"] for item in items], dtype=torch.long
            ),
            "memory_strength_before": torch.tensor(strengths_before, dtype=torch.float32),
            "buffer_indices": indices,
        }

    # ------------------------------------------------------------------ #
    # Capacity Optimization (60% buffer reduction enforcement)            #
    # ------------------------------------------------------------------ #
    def _evict(self, current_step: int) -> int:
        """Evict the least valuable samples until the buffer fits capacity.

        Eviction priority = S(t) / nll_surprise: the *least valuable* items are
        those with the highest Memory Strength S(t) (strong, already consolidated)
        AND the lowest original NLL surprise (familiar / low novelty).
        """
        overflow = len(self.buffer) - self.max_buffer_size
        if overflow <= 0:
            return 0
        priorities = [
            self.memory_strength(item, current_step) / (item["nll_surprise"] + 1e-8)
            for item in self.buffer
        ]
        evict_indices = sorted(
            sorted(
                ((priority, index) for index, priority in enumerate(priorities)),
                key=lambda pair: (pair[0], pair[1]),
                reverse=True,
            )[:overflow],
            key=lambda pair: pair[1],
            reverse=True,
        )
        for _, index in evict_indices:
            del self.buffer[index]
            self.stats["evicted"] += 1
        return len(evict_indices)

    def summary(self, current_step: int) -> dict[str, Any]:
        strengths = self.get_memory_strengths(current_step)
        return {
            "size": len(self.buffer),
            "capacity": self.max_buffer_size,
            "at_risk": sum(
                strength < self.decay_risk_threshold for strength in strengths
            ),
            "mean_strength": sum(strengths) / len(strengths) if strengths else 0.0,
        }
