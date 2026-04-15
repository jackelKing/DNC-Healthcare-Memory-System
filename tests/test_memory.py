import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from core.memory_matrix import MemoryMatrix
from reasoning.causal_graph import CausalGraph
from data.patients import PATIENTS


def test_memory_write_read():
    mem = MemoryMatrix(num_slots=16, vector_dim=8)
    vec = np.random.rand(8).astype(np.float32)
    vec /= np.linalg.norm(vec)
    slot = mem.write("E001", vec, {"condition": "Test"})
    assert 0 <= slot < 16
    assert mem.slots[slot].event_id == "E001"


def test_content_addressing():
    mem = MemoryMatrix(num_slots=16, vector_dim=8)
    vecs = [np.random.rand(8).astype(np.float32) for _ in range(4)]
    for i, v in enumerate(vecs):
        v /= np.linalg.norm(v)
        mem.write(f"E00{i}", v, {})
    weights = mem.content_address(vecs[2].copy())
    assert weights.shape == (16,)
    assert abs(weights.sum() - 1.0) < 1e-4


def test_causal_graph_build():
    cg = CausalGraph()
    cg.build_from_events(PATIENTS[0]["events"])
    assert cg.graph.number_of_nodes() == len(PATIENTS[0]["events"])
    assert cg.graph.number_of_edges() > 0


def test_causal_rank():
    cg = CausalGraph()
    cg.build_from_events(PATIENTS[0]["events"])
    ranked = cg.rank_causes("E006")
    if ranked:
        scores = [c["causal_score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)


def test_patient_data_integrity():
    for p in PATIENTS:
        assert "id" in p and "events" in p
        assert len(p["events"]) >= 5
        for e in p["events"]:
            assert all(k in e for k in ["id","timestamp","condition","causal_links"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
