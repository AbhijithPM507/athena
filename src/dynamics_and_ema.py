"""Pillar 2 & 3 - Parameter Trajectory Dynamics, Instability Ratio, EMA Consolidation.

Governs the interaction between the Fast (Plastic) LoRA pathway and the
Slow (Stable) LoRA pathway:

* Parameter Trajectory Distance (normalized Frobenius / L2 shift of the fast
  pathway away from its reference snapshot)::

      d_theta = sqrt(sum_p ||theta_F,p^(t) - theta_F,p^(ref)||^2)
              / (sqrt(sum_p ||theta_F,p^(ref)||^2) + eps)

* Step-to-step parameter velocity and Instability Ratio::

      v_t = ||theta_F^(t) - theta_F^(t-1)||_2
      r   = v_t / (mean(v_1..t) + eps)

* Adaptive consolidation weight (Slow pathway shields itself when the Fast
  pathway is volatile)::

      alpha_eff = alpha_base / (1 + gamma * r)   in (0, alpha_base]

* Dynamics-triggered EMA consolidation::

      theta_S <- (1 - alpha_eff) * theta_S + alpha_eff * theta_F
      theta_F^(ref) <- theta_F^(t)
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch

LORA_KEY_MARKERS = ("lora_a", "lora_b")


def is_lora_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in LORA_KEY_MARKERS)


def select_lora_tensors(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state_dict.items() if is_lora_key(key)}


def compute_trajectory_distance(
    fast_lora_state_dict: dict[str, Any],
    ref_lora_state_dict: dict[str, Any],
    epsilon: float = 1e-8,
) -> float:
    """Normalized Frobenius / L2 distance across all LoRA adapter tensors.

    d_theta = sqrt(sum_p ||theta_F,p^(t) - theta_F,p^(ref)||^2_2)
            / (sqrt(sum_p ||theta_F,p^(ref)||^2_2) + eps)
    """
    fast_lora = select_lora_tensors(fast_lora_state_dict)
    ref_lora = select_lora_tensors(ref_lora_state_dict)
    if not ref_lora:
        raise ValueError("reference state dict contains no LoRA (lora_A/lora_B) tensors")
    if set(fast_lora) != set(ref_lora):
        raise ValueError(
            "fast and reference LoRA tensor sets differ: "
            f"fast-only={sorted(set(fast_lora) - set(ref_lora))}, "
            f"ref-only={sorted(set(ref_lora) - set(fast_lora))}"
        )
    numerator_sq = 0.0
    denominator_sq = 0.0
    for key, ref_value in ref_lora.items():
        ref_tensor = ref_value.detach().float()
        fast_tensor = fast_lora[key].detach().float().to(ref_tensor.device)
        difference = fast_tensor - ref_tensor
        numerator_sq += float(torch.sum(difference * difference))
        denominator_sq += float(torch.sum(ref_tensor * ref_tensor))
    return math.sqrt(numerator_sq) / (math.sqrt(denominator_sq) + epsilon)


class ConsolidationResult(NamedTuple):
    """Uniform return type of ``step_consolidation`` for both branches."""

    triggered: bool
    slow_state_dict: dict[str, Any]
    delta_theta: float
    instability_ratio: float
    alpha_eff: float


class ParameterDynamicsAndConsolidation:
    """Trajectory dynamics monitor + instability-gated EMA consolidation."""

    def __init__(
        self,
        delta_theta_threshold: float = 0.15,
        ema_alpha_base: float = 0.20,
        instability_gamma: float = 2.0,
        epsilon: float = 1e-8,
        velocity_window: int | None = None,
    ) -> None:
        if delta_theta_threshold <= 0.0:
            raise ValueError("delta_theta_threshold must be > 0")
        if not 0.0 < ema_alpha_base <= 1.0:
            raise ValueError("ema_alpha_base must lie in (0, 1]")
        if instability_gamma < 0.0:
            raise ValueError("instability_gamma must be >= 0")
        self.delta_theta_threshold = float(delta_theta_threshold)
        self.ema_alpha_base = float(ema_alpha_base)
        self.instability_gamma = float(instability_gamma)
        self.epsilon = float(epsilon)
        if velocity_window is not None and velocity_window < 1:
            raise ValueError("velocity_window must be >= 1 or None")
        self.velocity_window = velocity_window

        self.reference_state_dict: dict[str, Any] | None = None
        self.previous_flat: torch.Tensor | None = None
        self.velocities: list[float] = []
        self.step_count = 0
        self.consolidation_count = 0

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> "ParameterDynamicsAndConsolidation":
        params = config.get("athena_hyperparams", config) if isinstance(config, dict) else {}
        allowed = {"delta_theta_threshold", "ema_alpha_base", "instability_gamma"}
        kwargs = {key: value for key, value in dict(params).items() if key in allowed}
        kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------ #
    # Reference snapshot                                                  #
    # ------------------------------------------------------------------ #
    def set_reference(self, fast_state_dict: dict[str, Any]) -> None:
        """Snapshot theta_F^(ref) <- theta_F (LoRA tensors only, deep clone)."""
        lora = select_lora_tensors(fast_state_dict)
        if not lora:
            raise ValueError("fast state dict contains no LoRA (lora_A/lora_B) tensors")
        self.reference_state_dict = {
            key: value.detach().cpu().clone() for key, value in lora.items()
        }

    def reset_dynamics(self) -> None:
        self.reference_state_dict = None
        self.previous_flat = None
        self.velocities = []
        self.step_count = 0
        self.consolidation_count = 0

    # ------------------------------------------------------------------ #
    # Velocity & Instability Ratio                                        #
    # ------------------------------------------------------------------ #
    def _flatten_lora(self, state_dict: dict[str, Any]) -> torch.Tensor:
        lora = select_lora_tensors(state_dict)
        if not lora:
            raise ValueError("state dict contains no LoRA (lora_A/lora_B) tensors")
        keys = sorted(lora)
        return torch.cat(
            [lora[key].detach().cpu().float().reshape(-1) for key in keys]
        )

    def track_velocity(self, fast_state_dict: dict[str, Any]) -> float | None:
        """v_t = ||theta_F^(t) - theta_F^(t-1)||_2; returns None on the first step."""
        current_flat = self._flatten_lora(fast_state_dict)
        velocity: float | None = None
        if self.previous_flat is not None and self.previous_flat.shape == current_flat.shape:
            velocity = float(torch.linalg.vector_norm(current_flat - self.previous_flat))
            self.velocities.append(velocity)
        self.previous_flat = current_flat.clone()
        return velocity

    def compute_instability_ratio(self) -> float:
        """r = v_t / (mean(v_1..t) + eps); 0.0 while no velocity history exists."""
        if not self.velocities:
            return 0.0
        if self.velocity_window is not None:
            window = self.velocities[-self.velocity_window :]
        else:
            window = self.velocities
        v_t = self.velocities[-1]
        baseline = sum(window) / len(window)
        return v_t / (baseline + self.epsilon)

    def effective_alpha(self, instability_ratio: float) -> float:
        """alpha_eff = alpha_base / (1 + gamma * r), guaranteed in (0, alpha_base]."""
        return self.ema_alpha_base / (1.0 + self.instability_gamma * instability_ratio)

    # ------------------------------------------------------------------ #
    # Dynamics-Triggered EMA Consolidation                                #
    # ------------------------------------------------------------------ #
    def step_consolidation(
        self,
        fast_state_dict: dict[str, Any],
        slow_state_dict: dict[str, Any],
    ) -> ConsolidationResult:
        """One step of model time: measure dynamics, maybe consolidate.

        * Computes the trajectory distance d_theta of the fast pathway against
          its reference snapshot and the current parameter velocity.
        * If ``d_theta <= delta_theta_threshold``: returns ``triggered=False``
          with ``slow_state_dict`` left untouched.
        * If ``d_theta > delta_theta_threshold``: every LoRA tensor of the slow
          pathway receives the EMA update

              theta_S <- (1 - alpha_eff) * theta_S + alpha_eff * theta_F

          and the reference snapshot is reset to the current fast pathway.
        """
        if self.reference_state_dict is None:
            self.set_reference(fast_state_dict)

        self.track_velocity(fast_state_dict)
        self.step_count += 1
        delta_theta = compute_trajectory_distance(
            fast_state_dict, self.reference_state_dict, self.epsilon
        )
        instability_ratio = self.compute_instability_ratio()
        alpha_eff = self.effective_alpha(instability_ratio)
        triggered = delta_theta > self.delta_theta_threshold

        if not triggered:
            return ConsolidationResult(
                False, slow_state_dict, delta_theta, instability_ratio, alpha_eff
            )

        fast_lora = select_lora_tensors(fast_state_dict)
        slow_lora = select_lora_tensors(slow_state_dict)
        if set(fast_lora) != set(slow_lora):
            raise ValueError(
                "fast and slow LoRA tensor sets differ: "
                f"fast-only={sorted(set(fast_lora) - set(slow_lora))}, "
                f"slow-only={sorted(set(slow_lora) - set(fast_lora))}"
            )
        for key, slow_value in slow_lora.items():
            fast_value = fast_lora[key].detach().to(
                device=slow_value.device, dtype=slow_value.dtype
            )
            slow_state_dict[key] = (
                (1.0 - alpha_eff) * slow_value + alpha_eff * fast_value
            )
        self.set_reference(fast_state_dict)
        self.consolidation_count += 1
        return ConsolidationResult(
            True, slow_state_dict, delta_theta, instability_ratio, alpha_eff
        )

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.step_count,
            "consolidations": self.consolidation_count,
            "velocity_history": len(self.velocities),
            "last_velocity": self.velocities[-1] if self.velocities else None,
        }
