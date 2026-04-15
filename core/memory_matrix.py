"""DNC-inspired N x W memory matrix with content addressing and temporal linkage."""
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemorySlot:
    slot_id: int
    event_id: str
    content_vector: np.ndarray
    usage: float = 0.0
    write_time: int = 0
    metadata: dict = field(default_factory=dict)


class MemoryMatrix:
    def __init__(self, num_slots: int = 64, vector_dim: int = 768):
        self.N = num_slots
        self.W = vector_dim
        self.matrix = np.zeros((num_slots, vector_dim), dtype=np.float32)
        self.usage = np.zeros(num_slots, dtype=np.float32)
        self.write_times = np.zeros(num_slots, dtype=np.int32)
        self.slots: List[Optional[MemorySlot]] = [None] * num_slots
        self._write_counter = 0
        self.link_matrix = np.zeros((num_slots, num_slots), dtype=np.float32)
        self.precedence = np.zeros(num_slots, dtype=np.float32)

    def _find_write_slot(self) -> int:
        empty = [i for i, s in enumerate(self.slots) if s is None]
        return empty[0] if empty else int(np.argmin(self.usage))

    def write(self, event_id: str, vector: np.ndarray, metadata: dict) -> int:
        slot_idx = self._find_write_slot()
        self._write_counter += 1
        prev_w = self.precedence.copy()
        self.link_matrix = (
            (1 - prev_w[:, None] - prev_w[None, :]) * self.link_matrix
            + prev_w[:, None] * np.eye(self.N)[slot_idx]
        )
        np.fill_diagonal(self.link_matrix, 0.0)
        self.precedence = (1 - np.sum(prev_w)) * prev_w
        self.precedence[slot_idx] = 1.0
        self.matrix[slot_idx] = vector
        self.usage[slot_idx] = 1.0
        self.write_times[slot_idx] = self._write_counter
        self.slots[slot_idx] = MemorySlot(
            slot_id=slot_idx, event_id=event_id,
            content_vector=vector.copy(), usage=1.0,
            write_time=self._write_counter, metadata=metadata
        )
        return slot_idx

    def content_address(self, query: np.ndarray, beta: float = 5.0) -> np.ndarray:
        norms = np.linalg.norm(self.matrix, axis=1) + 1e-9
        q_norm = np.linalg.norm(query) + 1e-9
        cosine = (self.matrix @ query) / (norms * q_norm)
        scores = beta * cosine - (beta * cosine).max()
        weights = np.exp(scores)
        return (weights / (weights.sum() + 1e-9)).astype(np.float32)

    def read(self, weights: np.ndarray) -> np.ndarray:
        return weights @ self.matrix

    def forward_weights(self, prev_read_weights: np.ndarray) -> np.ndarray:
        fwd = self.link_matrix.T @ prev_read_weights
        return (fwd / (fwd.sum() + 1e-9)).astype(np.float32)

    def backward_weights(self, prev_read_weights: np.ndarray) -> np.ndarray:
        bwd = self.link_matrix @ prev_read_weights
        return (bwd / (bwd.sum() + 1e-9)).astype(np.float32)

    def decay_usage(self, gamma: float = 0.99):
        self.usage *= gamma

    def get_top_slots(self, weights: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        idx = np.argsort(weights)[::-1][:top_k]
        return [(int(i), float(weights[i])) for i in idx if self.slots[i] is not None]

    def get_slot_by_event(self, event_id: str) -> Optional[int]:
        for i, s in enumerate(self.slots):
            if s is not None and s.event_id == event_id:
                return i
        return None

    def matrix_snapshot(self) -> np.ndarray:
        return self.matrix.copy()
