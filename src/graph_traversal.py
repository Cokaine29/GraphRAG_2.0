"""
graph_traversal.py — query-time graph traversal for multi-hop reasoning.

When a user asks something like "how does BERT relate to attention mechanisms?"
this module:
  1. Finds which graph nodes match entities in the query
  2. Traverses edges outward (BFS) up to N hops
  3. Returns the collected subgraph as readable text context for the LLM

This is the component that enables multi-hop reasoning — following chains of
relationships across entities that no single document chunk would contain.

Usage:
    from src.graph_traversal import GraphTraversal
    gt = GraphTraversal()
    gt.load_graph()
    context = gt.query("How does BERT relate to attention?")
"""

import json
import logging
from collections import deque
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Set, Optional

import networkx as nx

from src.config import cfg, GRAPH_PATH
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

__all__ = ["GraphTraversal"]

# ── Prompt for entity linking ─────────────────────────────────
ENTITY_LINK_PROMPT = """You are an entity extraction system for a knowledge graph query engine.

Given a user query, extract the key entities (concepts, models, algorithms, techniques)
that should be looked up in the knowledge graph.

User Query: "{query}"

Return ONLY a JSON array of entity names, no explanation:
["entity1", "entity2", ...]

Rules:
- Extract 1-4 most important entities
- Use the canonical name (e.g. "BERT" not "the BERT model")
- If no specific entities, return the 1-2 most important concepts"""

# ── Prompt for answer generation from subgraph ────────────────
TRAVERSAL_ANSWER_PROMPT = """You are answering a question using structured knowledge extracted
from a knowledge graph. The context below shows entities and their relationships,
retrieved via graph traversal.

Knowledge Graph Context:
{graph_context}

User Question: {query}

Instructions:
- Use the graph relationships to reason across multiple concepts
- Explicitly mention how concepts connect to each other
- If the graph context shows a chain A → B → C, explain that chain
- Be specific about relationship types when relevant

Answer:"""


class GraphTraversal:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.graph: Optional[nx.Graph] = None
        self.llm      = llm or LLMClient(purpose="generation")
        self.max_hops = 2        # how many hops to traverse from seed nodes
        self.max_nodes = 20      # cap on subgraph size to avoid prompt overflow
        self.sim_threshold = 0.6 # minimum name similarity to match a node

    # ── Graph loading ─────────────────────────────────────────

    def load_graph(self, path: Optional[str] = None) -> nx.Graph:
        gml_path = path or GRAPH_PATH
        if not gml_path.exists():
            raise FileNotFoundError(
                f"Graph not found at {gml_path}.\n"
                "Run knowledge graph construction first."
            )
        self.graph = nx.read_gml(str(gml_path))
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    # ── Main entry point ──────────────────────────────────────

    def query(self, query_text: str) -> Dict:
        """
        Full traversal pipeline: entity link → traverse → format → answer.

        Returns:
        {
            answer:        str,
            seed_entities: list of matched node names,
            subgraph_nodes: list of all traversed nodes,
            n_hops:        int,
            graph_context: str  (the raw context fed to LLM)
        }
        """
        if self.graph is None:
            raise RuntimeError("Call load_graph() first.")

        # Step 1: Extract entities from query
        seed_names = self._extract_entities(query_text)
        logger.info(f"Extracted entities: {seed_names}")

        # Step 2: Find matching nodes in graph
        seed_nodes = self._link_entities(seed_names)
        logger.info(f"Matched nodes: {seed_nodes}")

        if not seed_nodes:
            # Fallback: try keyword matching
            seed_nodes = self._keyword_fallback(query_text)
            logger.info(f"Fallback nodes: {seed_nodes}")

        # Step 3: BFS traversal from seed nodes
        subgraph_nodes, subgraph_edges = self._bfs_traverse(seed_nodes)
        logger.info(f"Subgraph: {len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges")

        # Step 4: Format subgraph as readable context
        graph_context = self._format_subgraph(
            subgraph_nodes, subgraph_edges, seed_nodes
        )

        # Step 5: Generate answer
        prompt = TRAVERSAL_ANSWER_PROMPT.format(
            graph_context=graph_context,
            query=query_text,
        )
        answer = self.llm.generate(prompt, temperature=0.3)

        return {
            "answer":         answer,
            "seed_entities":  seed_nodes,
            "subgraph_nodes": list(subgraph_nodes),
            "n_hops":         self.max_hops,
            "graph_context":  graph_context,
        }

    # ── Entity extraction from query ──────────────────────────

    def _extract_entities(self, query: str) -> List[str]:
        """Ask LLM to extract entity names from the query."""
        prompt = ENTITY_LINK_PROMPT.format(query=query)
        try:
            raw = self.llm.generate(prompt, temperature=0.0)
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    # ── Entity linking to graph nodes ────────────────────────

    def _link_entities(self, entity_names: List[str]) -> List[str]:
        """
        Match extracted entity names to actual nodes in the graph.
        Uses fuzzy string similarity to handle minor name differences.
        """
        graph_nodes = list(self.graph.nodes())
        matched = []

        for entity in entity_names:
            best_node  = None
            best_score = 0.0

            entity_lower = entity.lower()

            for node in graph_nodes:
                node_lower = node.lower()

                # Exact match — always take it
                if entity_lower == node_lower:
                    best_node  = node
                    best_score = 1.0
                    break

                # Substring match
                if entity_lower in node_lower or node_lower in entity_lower:
                    score = 0.85
                    if score > best_score:
                        best_score = score
                        best_node  = node

                # Fuzzy similarity
                score = SequenceMatcher(None, entity_lower, node_lower).ratio()
                if score > best_score and score >= self.sim_threshold:
                    best_score = score
                    best_node  = node

            if best_node:
                matched.append(best_node)

        return list(set(matched))  # deduplicate

    def _keyword_fallback(self, query: str) -> List[str]:
        """
        If entity linking fails, fall back to keyword matching
        against node names directly.
        """
        query_words = set(query.lower().split())
        matched = []
        for node in self.graph.nodes():
            node_words = set(node.lower().replace("_", " ").split())
            if query_words & node_words:  # any word overlap
                matched.append(node)
        return matched[:3]  # limit to top 3

    # ── BFS traversal ─────────────────────────────────────────

    def _bfs_traverse(
        self,
        seed_nodes: List[str]
    ) -> Tuple[Set[str], List[Dict]]:
        """
        Breadth-first search from seed nodes up to max_hops.

        Returns:
        - visited_nodes: set of all node names reached
        - collected_edges: list of edge dicts with source/target/relation
        """
        visited_nodes: Set[str]  = set(seed_nodes)
        collected_edges: List[Dict] = []
        queue = deque([(node, 0) for node in seed_nodes])

        while queue:
            current_node, depth = queue.popleft()

            if depth >= self.max_hops:
                continue

            if len(visited_nodes) >= self.max_nodes:
                break

            # Traverse outgoing edges (directed graph)
            for _, neighbor, edge_data in self.graph.out_edges(
                current_node, data=True
            ):
                edge_info = {
                    "source":        current_node,
                    "target":        neighbor,
                    "relation_type": edge_data.get("relation_type", "related_to"),
                    "description":   edge_data.get("description", ""),
                    "weight":        edge_data.get("computed_weight", 1.0),
                    "depth":         depth + 1,
                }
                collected_edges.append(edge_info)

                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

            # Also traverse incoming edges for richer context
            for neighbor, _, edge_data in self.graph.in_edges(
                current_node, data=True
            ):
                if neighbor not in visited_nodes:
                    edge_info = {
                        "source":        neighbor,
                        "target":        current_node,
                        "relation_type": edge_data.get("relation_type", "related_to"),
                        "description":   edge_data.get("description", ""),
                        "weight":        edge_data.get("computed_weight", 1.0),
                        "depth":         depth + 1,
                    }
                    collected_edges.append(edge_info)
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return visited_nodes, collected_edges

    # ── Subgraph formatting ───────────────────────────────────

    def _format_subgraph(
        self,
        nodes: Set[str],
        edges: List[Dict],
        seed_nodes: List[str]
    ) -> str:
        """
        Convert the traversed subgraph into readable text for the LLM.
        Seed nodes are marked as starting points.
        Edges are sorted by depth so the LLM sees direct connections first.
        """
        lines = []

        # Node descriptions
        lines.append("ENTITIES:")
        for node in sorted(nodes):
            node_data = self.graph.nodes.get(node, {})
            desc      = node_data.get("description", "")
            etype     = node_data.get("entity_type", "concept")
            marker    = " [QUERY ENTITY]" if node in seed_nodes else ""
            if desc:
                lines.append(f"  • {node} ({etype}){marker}: {desc[:150]}")
            else:
                lines.append(f"  • {node} ({etype}){marker}")

        # Relationships sorted by depth (direct connections first)
        lines.append("\nRELATIONSHIPS (ordered by distance from query entities):")
        sorted_edges = sorted(edges, key=lambda x: x["depth"])

        seen_edges = set()
        for edge in sorted_edges:
            edge_key = (edge["source"], edge["target"], edge["relation_type"])
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            hop_label = f"[hop {edge['depth']}]"
            rel_line  = f"  {hop_label} {edge['source']} --[{edge['relation_type']}]--> {edge['target']}"
            if edge["description"]:
                rel_line += f"\n         {edge['description'][:100]}"
            lines.append(rel_line)

        return "\n".join(lines)

    # ── Utility ───────────────────────────────────────────────

    def get_node_info(self, node_name: str) -> Dict:
        """Get all stored attributes for a node."""
        if node_name not in self.graph.nodes:
            return {}
        return dict(self.graph.nodes[node_name])

    def find_path(self, source: str, target: str) -> List[str]:
        """
        Find the shortest path between two named entities.
        Useful for explaining how two concepts are connected.
        """
        try:
            G_undirected = self.graph.to_undirected()
            return nx.shortest_path(G_undirected, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
