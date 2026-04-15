"""DNC Controller: orchestrates memory reads/writes for a patient event timeline."""
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from core.memory_matrix import MemoryMatrix
from ollama_client.client import embed_text, embed_batch

logger = logging.getLogger(__name__)


def _event_to_text(event: Dict[str, Any]) -> str:
    return (
        f"Condition: {event['condition']}. Year: {event['timestamp']}. "
        f"Symptoms: {', '.join(event.get('symptoms', []))}. "
        f"Treatments: {', '.join(event.get('treatments', []))}. "
        f"Outcomes: {', '.join(event.get('outcomes', []))}."
    )


class DNCController:
    def __init__(self, num_slots: int = 64, vector_dim: int = 768):
        self.memory = MemoryMatrix(num_slots=num_slots, vector_dim=vector_dim)
        self.event_index: Dict[str, Dict[str, Any]] = {}
        self.slot_index: Dict[str, int] = {}
        self._last_read_weights = np.zeros(num_slots, dtype=np.float32)

    def load_patient(self, patient: Dict[str, Any]):
        events = patient["events"]
        texts = [_event_to_text(e) for e in events]
        embeddings = embed_batch(texts)
        for event, vec in zip(events, embeddings):
            eid = event["id"]
            self.event_index[eid] = event
            slot = self.memory.write(
                event_id=eid, vector=vec,
                metadata={
                    "condition": event["condition"],
                    "timestamp": event["timestamp"],
                    "severity": event.get("severity", 0.5),
                    "causal_links": event.get("causal_links", []),
                }
            )
            self.slot_index[eid] = slot
        logger.info(f"Loaded {len(events)} events for patient {patient['id']}")

    def query_memory(
        self, query_text: str, top_k: int = 5, use_temporal: bool = True
    ) -> Tuple[List[Tuple[int, float]], np.ndarray, np.ndarray, np.ndarray]:
        q_vec = embed_text(query_text)
        content_w = self.memory.content_address(q_vec, beta=5.0)
        fwd_w = np.zeros_like(content_w)
        bwd_w = np.zeros_like(content_w)
        if use_temporal and self._last_read_weights.sum() > 0:
            fwd_w = self.memory.forward_weights(self._last_read_weights)
            bwd_w = self.memory.backward_weights(self._last_read_weights)
        combined = 0.7 * content_w + 0.15 * fwd_w + 0.15 * bwd_w
        combined /= combined.sum() + 1e-9
        self._last_read_weights = combined.copy()
        self.memory.decay_usage()
        return self.memory.get_top_slots(combined, top_k=top_k), content_w, fwd_w, bwd_w

    def get_event_by_slot(self, slot_idx: int) -> Dict[str, Any]:
        slot = self.memory.slots[slot_idx]
        if slot is None:
            return {}
        return self.event_index.get(slot.event_id, {})

    def reset_temporal_state(self):
        self._last_read_weights = np.zeros(self.memory.N, dtype=np.float32)
