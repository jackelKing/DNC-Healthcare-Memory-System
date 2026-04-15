"""PatientSession ties DNC controller, causal graph, and retrieval engine together."""
import logging
from typing import Dict, Any, Optional
from core.dnc_controller import DNCController
from reasoning.causal_graph import CausalGraph
from retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)
_session_cache: Dict[str, "PatientSession"] = {}


class PatientSession:
    def __init__(self, patient: Dict[str, Any], num_slots: int = 64, vector_dim: int = 768):
        self.patient = patient
        self.patient_id = patient["id"]
        self.controller = DNCController(num_slots=num_slots, vector_dim=vector_dim)
        self.causal_graph = CausalGraph()
        self.engine: Optional[RetrievalEngine] = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self.controller.load_patient(self.patient)
        self.causal_graph.build_from_events(self.patient["events"])
        self.engine = RetrievalEngine(self.controller, self.causal_graph)
        self._loaded = True

    def query(self, text: str, year_start: Optional[int] = None,
              year_end: Optional[int] = None, top_k: int = 5):
        self.load()
        self.controller.reset_temporal_state()
        return self.engine.retrieve(text, year_start=year_start, year_end=year_end, top_k=top_k)


def get_or_create_session(patient: Dict[str, Any]) -> PatientSession:
    pid = patient["id"]
    if pid not in _session_cache:
        _session_cache[pid] = PatientSession(patient)
    return _session_cache[pid]
