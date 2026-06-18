from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ActionTerm:
    name: str
    size: int
    low: float = 0.0
    high: float = 1.0


@dataclass(frozen=True)
class ObservationTerm:
    name: str
    size: int


class ActionSchema:
    """Named layout for policy actions."""

    def __init__(self, terms: Iterable[ActionTerm]):
        self.terms = tuple(terms)
        self.slices = self._make_slices(self.terms)
        self.dim = sum(term.size for term in self.terms)

    def split(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {term.name: actions[:, self.slices[term.name]] for term in self.terms}

    @staticmethod
    def _make_slices(terms: tuple[ActionTerm, ...]) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        start = 0
        for term in terms:
            stop = start + term.size
            slices[term.name] = slice(start, stop)
            start = stop
        return slices


class ObservationSchema:
    """Named layout for one frame of observations before history stacking."""

    def __init__(self, terms: Iterable[ObservationTerm]):
        self.terms = tuple(terms)
        self.slices = self._make_slices(self.terms)
        self.dim = sum(term.size for term in self.terms)

    def pack(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([values[term.name] for term in self.terms], dim=-1)

    @staticmethod
    def _make_slices(terms: tuple[ObservationTerm, ...]) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        start = 0
        for term in terms:
            stop = start + term.size
            slices[term.name] = slice(start, stop)
            start = stop
        return slices
