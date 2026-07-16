"""
router.py — hybrid query router using LLM zero-shot classification.

Classifies each query as:
  LOCAL   → specific fact, best answered by vector RAG
  GLOBAL  → thematic/synthesis, best answered by GraphRAG
  HYBRID  → needs both; contexts merged before generation

Usage:
    from src.router import QueryRouter, RouteType
    router = QueryRouter()
    decision = router.route("What are the main themes in this corpus?")
    print(decision.route_type, decision.confidence)
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

from src.config import cfg
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

__all__ = ["RouteType", "RouteDecision", "QueryRouter"]

ROUTER_PROMPT = """You are a query routing system for a GraphRAG-powered document chatbot.

Classify the following user query into one of three categories:

LOCAL  — The query asks for a specific fact, definition, detail, or quote from a
         particular document. Best answered by searching similar text passages.
         Examples: "What did paper X say about Y?", "Define attention mechanism",
                   "What year was BERT published?", "List the datasets used in paper Z"

GLOBAL — The query asks for synthesis, themes, comparisons, or high-level understanding
         across the entire corpus. Best answered using the knowledge graph communities.
         Examples: "What are the main research themes?", "How do these papers relate?",
                   "What is the overall contribution of this field?", "Summarize the corpus"

HYBRID — The query needs both specific facts AND broader context, or it is ambiguous.
         Examples: "Explain attention and how it fits into the transformer literature",
                   "What are the key findings, and how do they build on prior work?"

User Query: "{query}"

Respond ONLY with valid JSON — no explanation, no preamble:
{{
  "route_type": "LOCAL" | "GLOBAL" | "HYBRID",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining the classification>"
}}"""


class RouteType(str, Enum):
    LOCAL  = "LOCAL"
    GLOBAL = "GLOBAL"
    HYBRID = "HYBRID"


@dataclass
class RouteDecision:
    route_type:  RouteType
    confidence:  float
    reasoning:   str
    raw_query:   str


class QueryRouter:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm       = llm or LLMClient(purpose="generation")
        self.threshold = cfg.get("retrieval", {}).get("router_confidence_threshold", 0.6)

    def route(self, query: str) -> RouteDecision:
        """
        Classify a query and return a RouteDecision.
        Falls back to HYBRID if confidence < threshold or parsing fails.
        """
        prompt = ROUTER_PROMPT.format(query=query)

        try:
            raw = self.llm.generate(prompt, temperature=0.0)
            parsed = self._parse_json(raw)

            route_str  = parsed.get("route_type", "HYBRID").upper()
            confidence = float(parsed.get("confidence", 0.5))
            reasoning  = parsed.get("reasoning", "")

            # Validate route type
            if route_str not in [r.value for r in RouteType]:
                route_str = "HYBRID"

            # Low confidence → default to HYBRID
            if confidence < self.threshold:
                logger.info(f"Low confidence ({confidence:.2f}) → defaulting to HYBRID")
                route_str = "HYBRID"

            return RouteDecision(
                route_type=RouteType(route_str),
                confidence=confidence,
                reasoning=reasoning,
                raw_query=query,
            )

        except Exception as e:
            logger.error(f"Classification failed: {e}. Defaulting to HYBRID.")
            return RouteDecision(
                route_type=RouteType.HYBRID,
                confidence=0.5,
                reasoning=f"Fallback due to error: {e}",
                raw_query=query,
            )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)

