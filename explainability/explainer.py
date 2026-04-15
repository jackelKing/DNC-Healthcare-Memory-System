"""Explainability: structured JSON + human-readable + LLM narrative."""
import json
import logging
from typing import Dict, Any, List
import numpy as np
from retrieval.engine import RetrievalResult
from ollama_client.client import generate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical AI assistant. Given a patient query and structured causal memory "
    "results, provide a concise medically accurate explanation. Be factual and precise."
)


def build_explanation(result: RetrievalResult, use_llm: bool = True) -> Dict[str, Any]:
    structured = result.to_dict()
    structured["memory_debug"] = {
        "slots_accessed": [
            {
                "slot_index": s,
                "read_weight": round(w, 6),
                "content_weight": round(float(result.read_weights_content[s]), 6),
                "forward_weight": round(float(result.read_weights_forward[s]), 6),
                "backward_weight": round(float(result.read_weights_backward[s]), 6),
            }
            for s, w in result.memory_slots
        ],
        "top_content_weights": _top_n(result.read_weights_content, 5),
        "causal_path_length": len(result.causal_traversal_path),
    }
    rule_exp = _rule_explanation(result)
    structured["rule_explanation"] = rule_exp
    if use_llm:
        events_txt = "; ".join(
            f"{e.get('condition','')} ({e.get('timestamp','')})"
            for e in result.matched_events[:3]
        )
        causes_txt = "; ".join(
            f"{c.get('condition','')} -> {c.get('mechanism','')}"
            for c in result.ranked_causes[:4]
        )
        prompt = (
            f"Patient query: {result.query}\n\n"
            f"Events: {events_txt}\n\n"
            f"Causal factors: {causes_txt}\n\n"
            "Explain the causal chain clinically."
        )
        structured["llm_explanation"] = generate(prompt, system=SYSTEM_PROMPT)
    else:
        structured["llm_explanation"] = rule_exp
    return structured


def _rule_explanation(result: RetrievalResult) -> str:
    lines = [f"QUERY: {result.query}", ""]
    if result.matched_events:
        lines.append("RELEVANT CLINICAL EVENTS:")
        for ev in result.matched_events:
            conf = result.confidence_scores.get(ev.get("id", ""), 0)
            lines.append(
                f"  * {ev.get('condition','')} ({ev.get('timestamp','')}) "
                f"[confidence: {conf:.1%}]"
            )
            lines.append(f"    Symptoms: {', '.join(ev.get('symptoms', []))}")
            lines.append(f"    Outcomes: {', '.join(ev.get('outcomes', []))}")
    lines.append("")
    if result.ranked_causes:
        lines.append("CAUSAL CHAIN:")
        for i, c in enumerate(result.ranked_causes[:5], 1):
            lines.append(
                f"  {i}. {c.get('condition','')} "
                f"(hop={c.get('hop','?')}, score={c.get('causal_score',0):.3f})"
            )
            lines.append(f"     Mechanism: {c.get('mechanism','')}")
    if result.causal_traversal_path:
        lines += ["", "TRAVERSAL PATH: " + " -> ".join(result.causal_traversal_path)]
    if result.reasoning_chain:
        lines += ["", "REASONING CHAIN:"] + [f"  {s}" for s in result.reasoning_chain]
    return "\n".join(lines)


def _top_n(weights: np.ndarray, n: int) -> List[Dict[str, Any]]:
    if weights is None or len(weights) == 0:
        return []
    idx = np.argsort(weights)[::-1][:n]
    return [{"slot": int(i), "weight": round(float(weights[i]), 6)} for i in idx]


def explanation_to_json(explanation: Dict[str, Any]) -> str:
    def _s(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(type(obj))
    return json.dumps(explanation, indent=2, default=_s)