"""
community_detection.py - runs Leiden community detection on the knowledge graph.

Outputs:
- Community assignments at configured resolution levels
- Stats per level saved to outputs/graphs/

Usage:
    from src.community_detection import CommunityDetector
    cd = CommunityDetector()
    cd.load_graph()
    communities = cd.detect_all_levels()
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import networkx as nx

from src.config import cfg, GRAPH_PATH, ROOT

logger = logging.getLogger(__name__)

__all__ = ["CommunityDetector"]


class CommunityDetector:
    def __init__(self):
        self.resolutions   = cfg.get("community", {}).get("leiden_resolutions", [0.5, 1.0, 2.0])
        self.min_size      = cfg.get("community", {}).get("min_community_size", 3)
        self.graph: Optional[nx.Graph] = None
        self.communities: Dict[float, Dict[str, int]] = {}  # resolution ? {node: community_id}
        self._output_dir   = ROOT / "outputs" / "graphs"

    # -- Graph loading -----------------------------------------

    def load_graph(self, path: Optional[str] = None) -> nx.Graph:
        """Load the knowledge graph from GML file."""
        gml_path = Path(path) if path else GRAPH_PATH
        if not gml_path.exists():
            raise FileNotFoundError(f"Graph not found at {gml_path}. Run build_graph script first.")
        self.graph = nx.read_gml(str(gml_path))
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    # -- Leiden detection --------------------------------------

    def detect(self, resolution: float) -> Dict[str, int]:
        """
        Run Leiden at a given resolution.
        Returns {node_id: community_id} mapping.
        """
        if self.graph is None:
            raise RuntimeError("Call load_graph() first.")

        # Convert to undirected for community detection
        G_undirected = self.graph.to_undirected()

        # Use graspologic (preferred) with fallback to leidenalg
        try:
            partition = self._leiden_graspologic(G_undirected, resolution)
        except ImportError:
            partition = self._leiden_leidenalg(G_undirected, resolution)

        # Filter small communities
        partition = self._filter_small_communities(partition)

        n_communities = len(set(partition.values()))
        logger.info(f"Resolution {resolution}: {n_communities} communities (min_size={self.min_size})")
        return partition

    def _leiden_graspologic(self, G: nx.Graph, resolution: float) -> Dict[str, int]:
        from graspologic.partition import leiden
        node_list = list(G.nodes())
        adj = nx.to_numpy_array(G, nodelist=node_list, weight="computed_weight")
        labels = leiden(adj, resolution=resolution, random_seed=42)
        return {node_list[i]: int(labels[i]) for i in range(len(node_list))}

    def _leiden_leidenalg(self, G: nx.Graph, resolution: float) -> Dict[str, int]:
        import leidenalg
        import igraph as ig
        # Convert networkx ? igraph
        edges = [(u, v) for u, v in G.edges()]
        nodes = list(G.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        ig_edges = [(node_idx[u], node_idx[v]) for u, v in edges]
        weights  = [G[u][v].get("computed_weight", 1.0) for u, v in edges]

        ig_graph = ig.Graph(n=len(nodes), edges=ig_edges, directed=False)
        ig_graph.es["weight"] = weights

        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=42,
        )
        return {nodes[i]: partition.membership[i] for i in range(len(nodes))}

    def _filter_small_communities(self, partition: Dict[str, int]) -> Dict[str, int]:
        """Remove nodes belonging to communities smaller than min_size."""
        comm_counts = defaultdict(int)
        for comm_id in partition.values():
            comm_counts[comm_id] += 1

        valid_comms = {c for c, count in comm_counts.items() if count >= self.min_size}
        filtered = {node: comm for node, comm in partition.items() if comm in valid_comms}

        removed = len(partition) - len(filtered)
        if removed > 0:
            logger.info(f"Filtered {removed} nodes from small communities")
        return filtered

    # -- Detect all levels -------------------------------------

    def detect_all_levels(self) -> Dict[float, Dict[str, int]]:
        """Run Leiden at all configured resolutions."""
        for res in self.resolutions:
            self.communities[res] = self.detect(res)
            self._save_partition(res)
        return self.communities

    # -- Community member lookup -------------------------------

    def get_community_members(self, resolution: float) -> Dict[int, List[str]]:
        """
        Returns {community_id: [node1, node2, ...]} for a given resolution.
        """
        partition = self.communities.get(resolution)
        if not partition:
            raise ValueError(f"No partition for resolution={resolution}. Run detect() first.")

        members = defaultdict(list)
        for node, comm_id in partition.items():
            members[comm_id].append(node)
        return dict(members)

    def get_node_community(self, node: str, resolution: float = 1.0) -> int:
        """Look up which community a node belongs to."""
        return self.communities.get(resolution, {}).get(node, -1)

    # -- Node data helpers -------------------------------------

    def get_node_description(self, node: str) -> str:
        if self.graph and self.graph.has_node(node):
            return self.graph.nodes[node].get("description", "")
        return ""

    def get_community_edges(self, community_id: int, resolution: float) -> List[Dict]:
        """Get all edges between nodes within a community."""
        members = set(self.get_community_members(resolution).get(community_id, []))
        edges = []
        if self.graph:
            for u, v, data in self.graph.edges(data=True):
                if u in members and v in members:
                    edges.append({"source": u, "target": v, **data})
        return edges

    # -- Persistence -------------------------------------------

    def _save_partition(self, resolution: float) -> None:
        path = self._output_dir / f"partition_{resolution}.json"
        with open(path, "w") as f:
            json.dump(self.communities[resolution], f, indent=2)
        logger.info(f"Saved partition ? {path.name}")

    def load_partitions(self) -> None:
        """Load previously computed partitions from disk."""
        for res in self.resolutions:
            path = self._output_dir / f"partition_{res}.json"
            if path.exists():
                with open(path) as f:
                    self.communities[res] = json.load(f)
                logger.info(f"Loaded partition for resolution={res}")

    # -- Stats -------------------------------------------------

    def stats(self) -> Dict[float, Dict]:
        out = {}
        for res, partition in self.communities.items():
            members = self.get_community_members(res)
            sizes   = sorted([len(v) for v in members.values()], reverse=True)
            out[res] = {
                "n_communities": len(members),
                "largest":       sizes[0] if sizes else 0,
                "smallest":      sizes[-1] if sizes else 0,
                "avg_size":      round(sum(sizes)/len(sizes), 1) if sizes else 0,
            }
        return out

