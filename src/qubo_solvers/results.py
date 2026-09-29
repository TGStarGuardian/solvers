"""Results remain on the execution device until explicitly transferred."""

from dataclasses import dataclass, fields
from enum import IntEnum

import torch
from torch import Tensor


class TerminationReason(IntEnum):
    BUDGET_EXHAUSTED = 0
    LOCAL_OPTIMUM = 1


@dataclass(frozen=True, eq=False)
class OptimizationResult:
    """Assignments: (restarts, n); energies/counts/reasons: (restarts,).

    Histories: (samples, restarts), with matching (samples,) iteration indices.
    Iterations count sweeps for annealing and accepted flips for greedy search.
    Termination reasons are integer tensors encoded by TerminationReason.
    Assignment storage uses int8. In best-only mode final fields are None,
    best/diagnostic arrays have one row, and restart_indices holds its original
    index. Histories are unavailable in best-only mode.
    """

    final_assignments: Tensor | None
    final_energies: Tensor | None
    best_assignments: Tensor
    best_energies: Tensor
    iterations: Tensor
    termination_reasons: Tensor
    energy_history: Tensor | None = None
    history_iterations: Tensor | None = None
    restart_indices: Tensor | None = None

    @property
    def best_restart_index(self) -> Tensor:
        index = self.best_energies.argmin()
        return index if self.restart_indices is None else self.restart_indices[index]

    @property
    def best_assignment(self) -> Tensor:
        return self.best_assignments[self.best_energies.argmin()]

    @property
    def best_energy(self) -> Tensor:
        return self.best_energies.min()

    def to(self, *, device=None) -> "OptimizationResult":
        """Transfer all result tensors, preserving their dtypes."""
        return OptimizationResult(**{
            field.name: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for field in fields(self)
            for value in (getattr(self, field.name),)
        })
