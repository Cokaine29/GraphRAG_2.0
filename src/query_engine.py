import logging
from typing import Dict, Optional, Any
from src.config import cfg
from src.llm_client import LLMClient
from src.router import QueryRouter, RouteType
from src.vector_store import VectorStore
from src.graph_query import GraphQueryEngine

logger = logging.getLogger(__name__)

__all__ = ["QueryEngine"]

VECTOR_PROMPT = """Answer using only the provided context.

Context:
{context}

Question: {query}

Answer:"""

HYBRID_PROMPT = """Answer using all three sources below.

--- Document passages ---
{vector_context}

--- Knowledge graph relationships ---
{graph_context}

--- Thematic context ---
{community_context}

Question: {query}

Answer:"""


class QueryEngine:
    def __init__(self, vector_store: Optional[VectorStore] = None, graph_engine: Optional[GraphQueryEngine] = None,
                 graph_traversal: Optional[Any] = None, router: Optional[QueryRouter] = None, llm: Optional[LLMClient] = None):
        self.vector_store    = vector_store or VectorStore()
        self.graph_engine    = graph_engine
        self.graph_traversal = graph_traversal
        self.router          = router or QueryRouter()
        self.llm             = llm or LLMClient(purpose="generation")

    def query(self, query_text: str, force_route: Optional[str] = None) -> Dict[str, Any]:
        if force_route:
            from src.router import RouteDecision
            route_map = {
                "LOCAL":  RouteType.LOCAL,
                "GLOBAL": RouteType.GLOBAL,
                "HYBRID": RouteType.HYBRID,
            }
            decision = RouteDecision(
                route_type=route_map.get(force_route.upper(), RouteType.HYBRID),
                confidence=1.0, reasoning="Forced", raw_query=query_text,
            )
            use_traversal = "TRAVERSAL" in force_route.upper()
        else:
            decision      = self.router.route(query_text)
            use_traversal = (decision.route_type == RouteType.LOCAL and
                             self._looks_like_multihop(query_text))

        logger.info(f"Route: {decision.route_type.value} (confidence: {decision.confidence:.2f})")

        if decision.route_type == RouteType.GLOBAL:
            return self._run_graph_global(query_text, decision)
        elif decision.route_type == RouteType.LOCAL:
            if use_traversal and self.graph_traversal:
                return self._run_traversal(query_text, decision)
            return self._run_vector(query_text, decision)
        else:
            return self._run_hybrid(query_text, decision)

    def _looks_like_multihop(self, query: str) -> bool:
        signals = ["how does", "relate", "relationship", "connect", "connection",
                   "between", "through", "via", "path", "chain", "link",
                   "influence", "depend on", "what connects"]
        return any(s in query.lower() for s in signals)

    def _run_vector(self, query: str, decision: Any) -> Dict[str, Any]:
        chunks  = self.vector_store.query(query)
        context = "\n\n---\n\n".join(
            [f"[{c['doc_id']}]\n{c['text']}" for c in chunks])
        answer  = self.llm.generate(
            VECTOR_PROMPT.format(context=context, query=query), temperature=0.3)
        return {"answer": answer, "route": "LOCAL-VECTOR",
                "confidence": decision.confidence, "reasoning": decision.reasoning,
                "sources": chunks, "metadata": {"pipeline": "vector_rag"}}

    def _run_traversal(self, query: str, decision: Any) -> Dict[str, Any]:
        if not self.graph_traversal:
            return self._run_vector(query, decision)
        result = self.graph_traversal.query(query)
        return {"answer": result["answer"], "route": "LOCAL-TRAVERSAL",
                "confidence": decision.confidence, "reasoning": decision.reasoning,
                "sources": result["subgraph_nodes"],
                "metadata": {"pipeline": "graph_traversal",
                             "seed_entities": result["seed_entities"]}}

    def _run_graph_global(self, query: str, decision: Any) -> Dict[str, Any]:
        if not self.graph_engine:
            return self._run_vector(query, decision)
        result = self.graph_engine.query(query)
        return {"answer": result["answer"], "route": "GLOBAL",
                "confidence": decision.confidence, "reasoning": decision.reasoning,
                "sources": result.get("sources", []),
                "metadata": {"pipeline": "graphrag_map_reduce"}}

    def _run_hybrid(self, query: str, decision: Any) -> Dict[str, Any]:
        chunks = self.vector_store.query(query)
        vector_context = "\n\n---\n\n".join(
            [f"[{c['doc_id']}]\n{c['text']}" for c in chunks])
        graph_context, traversal_sources = "", []
        if self.graph_traversal:
            tr = self.graph_traversal.query(query)
            graph_context     = tr["graph_context"]
            traversal_sources = tr["subgraph_nodes"]
        community_context = ""
        if self.graph_engine:
            community_context = self.graph_engine.query(query)["answer"]
        answer = self.llm.generate(
            HYBRID_PROMPT.format(
                vector_context=vector_context or "None.",
                graph_context=graph_context or "None.",
                community_context=community_context or "None.",
                query=query), temperature=0.3)
        return {"answer": answer, "route": "HYBRID",
                "confidence": decision.confidence, "reasoning": decision.reasoning,
                "sources": {"vector": chunks, "traversal": traversal_sources},
                "metadata": {"pipeline": "hybrid"}}
