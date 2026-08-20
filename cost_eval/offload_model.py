"""Independent HBM/host policies for parameter and optimizer state offload."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OffloadPolicy:
    parameter: bool = False
    gradient: bool = False
    optimizer: bool = False

    @classmethod
    def from_parallel(cls, pc) -> "OffloadPolicy":
        return cls(
            bool(pc.parameter_offload),
            bool(pc.gradient_offload),
            bool(pc.optimizer_offload),
        )

    def split(self, breakdown):
        cls = breakdown.__class__
        resident = cls(
            0 if self.parameter else breakdown.parameter,
            0 if self.gradient else breakdown.gradient,
            0 if self.optimizer else breakdown.master_weight,
            0 if self.optimizer else breakdown.optimizer_state,
        )
        host = cls(
            breakdown.parameter if self.parameter else 0,
            breakdown.gradient if self.gradient else 0,
            breakdown.master_weight if self.optimizer else 0,
            breakdown.optimizer_state if self.optimizer else 0,
        )
        return resident, host
