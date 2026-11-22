from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import torch

@dataclass
class KnowledgeGraph:
    entities: list[str]
    relations: list[str]
    triples: list[tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.ent_index = {n: i for i, n in enumerate(self.entities)}
        self.rel_index = {n: i for i, n in enumerate(self.relations)}
        self._version = 0

    def add(self, h: str, r: str, t: str):
        self.triples.append((self.ent_index[h], self.rel_index[r], self.ent_index[t]))
        self._version += 1

    def set(self, h: str, r: str, t: str):
        hi, ri = (self.ent_index[h], self.rel_index[r])
        self.triples = [x for x in self.triples if not (x[0] == hi and x[1] == ri)]
        self.add(h, r, t)

    def tails(self, h: str, r: str) -> list[str]:
        hi, ri = (self.ent_index[h], self.rel_index[r])
        return [self.entities[t] for a, b, t in self.triples if a == hi and b == ri]

class ExactScorer:

    def __init__(self, kg: KnowledgeGraph, max_tails: int=4, device: str | torch.device='cpu'):
        self.kg, self.M, self.device = (kg, max_tails, torch.device(device))
        self._built_version = -1
        self._build()

    def _build(self):
        R = len(self.kg.relations)
        t = torch.tensor(self.kg.triples, dtype=torch.long).view(-1, 3)
        keys = t[:, 0] * R + t[:, 1]
        order = torch.argsort(keys, stable=True)
        self.keys = keys[order].to(self.device)
        self.tails_flat = t[order, 2].to(self.device)
        self.R = R
        self._built_version = self.kg._version

    @torch.no_grad()
    def __call__(self, h: torch.Tensor, r: torch.Tensor):
        if self._built_version != self.kg._version:
            self._build()
        P, M = (h.numel(), self.M)
        if self.keys.numel() == 0:
            return (torch.zeros(P, M, dtype=torch.long, device=h.device), torch.zeros(P, M, device=h.device))
        q = h * self.R + r
        start = torch.searchsorted(self.keys, q)
        end = torch.searchsorted(self.keys, q, right=True)
        cnt = (end - start).clamp(max=M)
        ar = torch.arange(M, device=h.device)
        idx = (start[:, None] + ar).clamp(max=self.keys.numel() - 1)
        mask = ar[None, :] < cnt[:, None]
        tails = torch.where(mask, self.tails_flat[idx], torch.zeros_like(idx))
        probs = mask.float() / cnt.clamp(min=1)[:, None].float()
        return (tails, probs)

    @torch.no_grad()
    def has_rel(self) -> torch.Tensor | None:
        if self._built_version != self.kg._version:
            self._build()
        E, R = (len(self.kg.entities), self.R)
        if E * R > 1 << 28:
            return None
        if getattr(self, '_has_version', None) != self._built_version:
            has = torch.zeros(E * R, dtype=torch.bool, device=self.device)
            has[self.keys] = True
            self._has, self._has_version = (has.view(E, R), self._built_version)
        return self._has

@dataclass
class GraphTensors:
    ent_keys: torch.Tensor
    rel_keys: torch.Tensor
    name_tok: torch.Tensor
    name_len: torch.Tensor
    scorer: Callable
    kg: KnowledgeGraph | None = None
    rel_keys_first: torch.Tensor | None = None
    alias: dict | None = None
    rel_alias: dict | None = None
