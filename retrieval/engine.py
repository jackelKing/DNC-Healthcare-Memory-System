"""Retrieval engine: semantic search + temporal filter + causal traversal + confidence."""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from core.dnc_controller import DNCController
from reasoning.causal_graph import CausalGraph
from ollama_client.client import embed_text

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class RetrievalResult:
    def __init__(self, query, matched_events, ranked_causes, reasoning_chain,
                 confidence_scores, memory_slots, read_weights_content,
                 read_weights_forward, read_weights_backward, causal_traversal_path):
        self.query = query
        self.matched_events = matched_events
        self.ranked_causes = ranked_causes
        self.reasoning_chain = reasoning_chain
        self.confidence_scores = confidence_scores
        self.memory_slots = memory_slots
        self.read_weights_content = read_weights_content
        self.read_weights_forward = read_weights_forward
        self.read_weights_backward = read_weights_backward
        self.causal_traversal_path = causal_traversal_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "matched_events": self.matched_events,
            "ranked_causes": self.ranked_causes,
            "reasoning_chain": self.reasoning_chain,
            "confidence_scores": self.confidence_scores,
            "memory_slots": [(int(s), float(w)) for s, w in self.memory_slots],
            "causal_traversal_path": self.causal_traversal_path,
        }


class RetrievalEngine:
    def __init__(self, controller: DNCController, causal_graph: CausalGraph):
        self.controller = controller
        self.causal_graph = causal_graph

    def retrieve(self, query: str, year_start: Optional[int] = None,
                 year_end: Optional[int] = None, top_k: int = 5) -> RetrievalResult:
        top_slots, cw, fw, bw = self.controller.query_memory(query, top_k=top_k * 2)
        matched_events = []
        for slot_idx, weight in top_slots:
            event = self.controller.get_event_by_slot(slot_idx)
            if not event:
                continue
            ts = event.get("timestamp", 0)
            if year_start and ts < year_start:
                continue
            if year_end and ts > year_end:
                continue
            matched_events.append({**event, "_slot": slot_idx, "_weight": float(weight)})
        matched_events = matched_events[:top_k]

        ranked_causes, traversal_path = [], []
        for ev in matched_events[:2]:
            eid = ev.get("id", "")
            causes = self.causal_graph.rank_causes(eid)
            ranked_causes.extend(causes)
            for c in causes[:2]:
                path = self.causal_graph.causal_path(c["event_id"], eid)
                if path:
                    traversal_path.extend(path)

        seen, unique_causes = set(), []
        for c in sorted(ranked_causes, key=lambda x: x.get("causal_score", 0), reverse=True):
            if c["event_id"] not in seen:
                seen.add(c["event_id"])
                unique_causes.append(c)

        traversal_path = list(dict.fromkeys(traversal_path))
        q_vec = embed_text(query)
        confidence = {}
        for ev in matched_events:
            eid = ev.get("id", "")
            si = self.controller.slot_index.get(eid)
            if si is not None:
                sim = _cosine(q_vec, self.controller.memory.matrix[si])
                confidence[eid] = round(max(0.0, min(1.0, sim)), 4)

        chain = [f"Query: {query}"]
        if matched_events:
            chain.append(f"Memory retrieved {len(matched_events)} event(s): " +
                         ", ".join(e.get("condition","") for e in matched_events))
        if unique_causes:
            chain.append("Causal analysis found upstream causes:")
            for i, c in enumerate(unique_causes[:4], 1):
                chain.append(f"  {i}. [{c.get('hop','?')}-hop] {c.get('condition','')} "
                             f"(score={c.get('causal_score',0):.3f}): {c.get('mechanism','')}")
        else:
            chain.append("No upstream causal chain identified.")

        return RetrievalResult(
            query=query, matched_events=matched_events, ranked_causes=unique_causes,
            reasoning_chain=chain, confidence_scores=confidence,
            memory_slots=[(ev["_slot"], ev["_weight"]) for ev in matched_events],
            read_weights_content=cw, read_weights_forward=fw, read_weights_backward=bw,
            causal_traversal_path=traversal_path
        )
