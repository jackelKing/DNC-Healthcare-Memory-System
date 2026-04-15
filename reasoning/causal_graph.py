"""Causal DAG built from explicit causal links. Multi-hop traversal, PageRank ranking."""
import logging
from typing import List, Dict, Any, Optional
import networkx as nx

logger = logging.getLogger(__name__)


class CausalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_events(self, events: List[Dict[str, Any]]):
        self.graph.clear()
        for e in events:
            self.graph.add_node(
                e["id"], condition=e["condition"], timestamp=e["timestamp"],
                severity=e.get("severity", 0.5), symptoms=e.get("symptoms", []),
                treatments=e.get("treatments", []), outcomes=e.get("outcomes", [])
            )
        for e in events:
            for link in e.get("causal_links", []):
                if self.graph.has_node(link["cause"]):
                    self.graph.add_edge(link["cause"], e["id"], mechanism=link.get("mechanism", ""))

    def get_direct_causes(self, event_id: str) -> List[Dict[str, Any]]:
        if not self.graph.has_node(event_id):
            return []
        return [
            {"event_id": p, "condition": self.graph.nodes[p].get("condition",""),
             "mechanism": self.graph.edges[p, event_id].get("mechanism",""),
             "timestamp": self.graph.nodes[p].get("timestamp", 0),
             "severity": self.graph.nodes[p].get("severity", 0.5), "hop": 1}
            for p in self.graph.predecessors(event_id)
        ]

    def get_all_causes(self, event_id: str, max_hops: int = 5) -> List[Dict[str, Any]]:
        if not self.graph.has_node(event_id):
            return []
        visited, results = set(), []

        def dfs(node: str, hop: int, path: List[str]):
            if hop > max_hops or node in visited:
                return
            visited.add(node)
            for pred in self.graph.predecessors(node):
                pn = self.graph.nodes[pred]
                results.append({
                    "event_id": pred, "condition": pn.get("condition",""),
                    "mechanism": self.graph.edges[pred, node].get("mechanism",""),
                    "timestamp": pn.get("timestamp", 0), "severity": pn.get("severity", 0.5),
                    "hop": hop, "path": path + [pred]
                })
                dfs(pred, hop + 1, path + [pred])

        dfs(event_id, 1, [event_id])
        return sorted(results, key=lambda x: (x["hop"], x["timestamp"]))

    def get_downstream_effects(self, event_id: str, max_hops: int = 5) -> List[Dict[str, Any]]:
        if not self.graph.has_node(event_id):
            return []
        visited, results = set(), []

        def dfs(node: str, hop: int, path: List[str]):
            if hop > max_hops or node in visited:
                return
            visited.add(node)
            for succ in self.graph.successors(node):
                sn = self.graph.nodes[succ]
                results.append({
                    "event_id": succ, "condition": sn.get("condition",""),
                    "mechanism": self.graph.edges[node, succ].get("mechanism",""),
                    "timestamp": sn.get("timestamp", 0), "severity": sn.get("severity", 0.5),
                    "hop": hop, "path": path + [succ]
                })
                dfs(succ, hop + 1, path + [succ])

        dfs(event_id, 1, [event_id])
        return sorted(results, key=lambda x: (x["hop"], x["timestamp"]))

    def rank_causes(self, event_id: str) -> List[Dict[str, Any]]:
        causes = self.get_all_causes(event_id)
        if not causes:
            return []
        try:
            pr = nx.pagerank(self.graph, alpha=0.85)
        except Exception:
            pr = {n: 1.0 for n in self.graph.nodes}
        for c in causes:
            hop_penalty = 1.0 / c["hop"]
            c["causal_score"] = round(0.4 * hop_penalty + 0.3 * c.get("severity", 0.5) + 0.3 * pr.get(c["event_id"], 0.0), 4)
        return sorted(causes, key=lambda x: x["causal_score"], reverse=True)

    def causal_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        try:
            return nx.shortest_path(self.graph, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_graph_data(self) -> Dict[str, Any]:
        return {
            "nodes": [{"id": n, **d} for n, d in self.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.graph.edges(data=True)]
        }
