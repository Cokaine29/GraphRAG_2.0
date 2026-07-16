"""
graph_query.py — GraphRAG query pipeline using map-reduce over community summaries.

Map step:  for each community summary, generate a partial answer + relevance score
Reduce step: aggregate top-k partial answers into a final coherent response

Usage:
    from src.graph_query import GraphQueryEngine
    engine = GraphQueryEngine()
    answer = engine.query("What are the main research themes in this corpus?")
"""

import json
import logging
from typing import List, Dict, Tuple, Optional

from src.config import cfg
from src.llm_client import LLMClient
from src.community_summarizer import CommunitySummarizer

logger = logging.getLogger(__name__)

__all__ = ["GraphQueryEngine"]

MAP_PROMPT = """You are analyzing a research corpus. Given the following community summary and a user query,
generate a partial answer if this community is relevant to the query.

Community Summary:
{summary}

User Query: {query}

Instructions:
- If this community is NOT relevant to the query, respond with: {{"relevance": 0, "answer": ""}}
- If relevant, provide a partial answer and rate relevance 1-10
- Respond ONLY with valid JSON: {{"relevance": <int 0-10>, "answer": "<partial answer>"}}

JSON Response:"""

REDUCE_PROMPT = """You are synthesizing multiple partial answers about a research corpus into a final comprehensive response.

User Query: {query}

Partial answers (ranked by relevance):
{partial_answers}

Instructions:
- Synthesize these partial answers into one coherent, well-structured response
- Cover all important points without repetition
- Use clear academic language
- If partial answers conflict, acknowledge both perspectives
- Length: 200-400 words

Final Answer:"""


class GraphQueryEngine:
    def __init__(self, summarizer: Optional[CommunitySummarizer] = None, llm: Optional[LLMClient] = None):
        self.summarizer       = summarizer
        self.llm              = llm or LLMClient(purpose="generation")
        self.top_communities  = cfg.get("retrieval", {}).get("top_communities", 10)
        self.map_top_k        = cfg.get("retrieval", {}).get("map_top_k", 5)
        self._summaries_cache: Dict[float, Dict] = {}

    # ── Main query ────────────────────────────────────────────

    def query(self, query_text: str, resolution: float = 1.0) -> Dict:
        """
        Run full map-reduce query.
        Returns {answer, sources, resolution, n_communities_used}
        """
        summaries = self._get_summaries(resolution)
        if not summaries:
            return {"answer": "No community summaries available. Run summarization first.",
                    "sources": [], "resolution": resolution}

        # MAP step
        logger.info(f"Map step: scoring {len(summaries)} communities...")
        partial_answers = self._map_step(query_text, summaries)

        # Filter zero-relevance and sort
        partial_answers = [(score, ans, cid) for score, ans, cid in partial_answers if score > 0]
        partial_answers.sort(key=lambda x: x[0], reverse=True)
        top_answers = partial_answers[:self.map_top_k]

        if not top_answers:
            return {"answer": "No relevant community summaries found for this query.",
                    "sources": [], "resolution": resolution}

        logger.info(f"Reduce step: aggregating {len(top_answers)} partial answers...")
        final_answer = self._reduce_step(query_text, top_answers)

        return {
            "answer":               final_answer,
            "resolution":           resolution,
            "n_communities_scored": len(summaries),
            "n_communities_used":   len(top_answers),
            "sources": [
                {"community_id": cid, "relevance": score,
                 "members": summaries[cid].get("members", [])}
                for score, _, cid in top_answers
            ],
        }

    # ── Map step ──────────────────────────────────────────────

    def _map_step(self, query: str, summaries: Dict) -> List[Tuple[int, str, int]]:
        """Returns list of (relevance_score, partial_answer, community_id)."""
        results = []
        items = list(summaries.items())[:self.top_communities]

        for comm_id, summary_data in items:
            summary_text = summary_data.get("summary", "")
            if not summary_text:
                continue

            prompt = MAP_PROMPT.format(
                summary=summary_text[:1500],  # cap to avoid context overflow
                query=query,
            )

            try:
                raw = self.llm.generate(prompt, temperature=0.0)
                parsed = self._parse_json(raw)
                relevance = int(parsed.get("relevance", 0))
                answer    = parsed.get("answer", "").strip()
                results.append((relevance, answer, comm_id))
            except Exception as e:
                logger.error(f"Map failed for community {comm_id}: {e}")
                results.append((0, "", comm_id))

        return results

    # ── Reduce step ───────────────────────────────────────────

    def _reduce_step(self, query: str, top_answers: List[Tuple]) -> str:
        """Aggregate top partial answers into a final response."""
        formatted = []
        for rank, (score, answer, cid) in enumerate(top_answers, 1):
            formatted.append(f"[{rank}] (relevance: {score}/10)\n{answer}")

        prompt = REDUCE_PROMPT.format(
            query=query,
            partial_answers="\n\n".join(formatted),
        )

        return self.llm.generate(prompt, temperature=0.3)

    # ── Helpers ───────────────────────────────────────────────

    def _get_summaries(self, resolution: float) -> Dict:
        if resolution not in self._summaries_cache:
            if self.summarizer:
                self._summaries_cache[resolution] = self.summarizer.load_summaries(resolution)
            else:
                raise RuntimeError("No summarizer provided. Initialize with a CommunitySummarizer.")
        return self._summaries_cache[resolution]

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM output, handling common formatting issues."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)

