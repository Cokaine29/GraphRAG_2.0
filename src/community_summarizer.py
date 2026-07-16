"""
community_summarizer.py — generates LLM summaries for each detected community.

Each summary captures the thematic content of a community in <=500 tokens.
All results are cached — safe to re-run if interrupted.

Usage:
    from src.community_summarizer import CommunitySummarizer
    cs = CommunitySummarizer(detector, llm)
    summaries = cs.summarize_all(resolution=1.0)
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.config import cfg, SUMMARIES_DIR
from src.llm_client import LLMClient
from src.community_detection import CommunityDetector

logger = logging.getLogger(__name__)

__all__ = ["CommunitySummarizer"]

SUMMARY_PROMPT = """You are analyzing a cluster of related concepts from a knowledge graph built from academic documents.

Community members (entities):
{entities}

Relationships within this community:
{relationships}

Write a concise thematic summary (maximum 400 words) that:
1. Identifies the central theme or topic of this community
2. Describes how the key concepts relate to each other
3. Notes any important sub-themes
4. Is written in clear academic language

Summary:"""


class CommunitySummarizer:
    def __init__(self, detector: CommunityDetector, llm: Optional[LLMClient] = None):
        self.detector  = detector
        self.llm       = llm or LLMClient(purpose="generation")
        self.delay     = cfg.get("extraction", {}).get("batch_delay_seconds", 1.5)
        self._summaries: Dict[str, Dict] = {}  # key: f"{resolution}_{community_id}"

    # ── Main entry point ──────────────────────────────────────

    def summarize_all(self, resolution: float = 1.0) -> Dict[int, Dict]:
        """
        Generate summaries for all communities at a given resolution.
        Returns {community_id: summary_dict}
        """
        members_map = self.detector.get_community_members(resolution)
        logger.info(f"Summarizing {len(members_map)} communities at resolution={resolution}")

        results = {}
        for i, (comm_id, members) in enumerate(members_map.items()):
            cache_key = self._cache_key(resolution, comm_id, members)
            cached    = self._load_from_cache(resolution, comm_id, cache_key)

            if cached:
                results[comm_id] = cached
                logger.info(f"Community {comm_id} — loaded from cache")
            else:
                logger.info(f"Community {comm_id} ({len(members)} nodes) — generating... [{i+1}/{len(members_map)}]")
                summary = self._summarize_community(comm_id, members, resolution)
                results[comm_id] = summary
                self._save_to_cache(resolution, comm_id, cache_key, summary)
                time.sleep(self.delay)

        self._save_all_summaries(resolution, results)
        return results

    def summarize_all_levels(self) -> Dict[float, Dict[int, Dict]]:
        """Run summarization for all resolution levels."""
        all_summaries = {}
        for res in cfg.get("community", {}).get("leiden_resolutions", [0.5, 1.0, 2.0]):
            if res in self.detector.communities:
                all_summaries[res] = self.summarize_all(resolution=res)
        return all_summaries

    # ── Single community ──────────────────────────────────────

    def _summarize_community(self, comm_id: int, members: List[str], resolution: float) -> Dict:
        # Build entity descriptions
        entity_lines = []
        for node in members:
            desc = self.detector.get_node_description(node)
            if desc:
                entity_lines.append(f"- {node}: {desc[:200]}")
            else:
                entity_lines.append(f"- {node}")

        # Build relationship descriptions
        edges = self.detector.get_community_edges(comm_id, resolution)
        rel_lines = []
        for edge in edges[:30]:  # cap at 30 to avoid prompt overflow
            rel_str = f"- {edge['source']} → {edge['target']}"
            if "relation_type" in edge:
                rel_str += f" [{edge['relation_type']}]"
            if "description" in edge:
                rel_str += f": {edge['description'][:100]}"
            rel_lines.append(rel_str)

        prompt = SUMMARY_PROMPT.format(
            entities="\n".join(entity_lines) or "No descriptions available",
            relationships="\n".join(rel_lines) or "No relationships found",
        )

        summary_text = self.llm.generate(prompt, temperature=0.3)

        return {
            "community_id":  comm_id,
            "resolution":    resolution,
            "n_members":     len(members),
            "members":       members,
            "n_edges":       len(edges),
            "summary":       summary_text,
        }

    # ── Cache ─────────────────────────────────────────────────

    def _cache_key(self, resolution: float, comm_id: int, members: List[str]) -> str:
        content = f"{resolution}_{comm_id}_{'_'.join(sorted(members))}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _cache_path(self, resolution: float, comm_id: int) -> Path:
        return SUMMARIES_DIR / f"summary_r{resolution}_c{comm_id}.json"

    def _load_from_cache(self, resolution: float, comm_id: int, expected_key: str) -> Optional[Dict]:
        path = self._cache_path(resolution, comm_id)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if data.get("cache_key") == expected_key:
                return data
        return None

    def _save_to_cache(self, resolution: float, comm_id: int, cache_key: str, summary: Dict) -> None:
        path = self._cache_path(resolution, comm_id)
        summary["cache_key"] = cache_key
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def _save_all_summaries(self, resolution: float, summaries: Dict[int, Dict]) -> None:
        """Save all summaries for a resolution level as one file."""
        path = SUMMARIES_DIR / f"all_summaries_r{resolution}.json"
        with open(path, "w") as f:
            json.dump(summaries, f, indent=2, default=str)
        logger.info(f"Saved {len(summaries)} summaries → {path.name}")

    # ── Loading summaries ─────────────────────────────────────

    def load_summaries(self, resolution: float) -> Dict[int, Dict]:
        """Load pre-computed summaries for a resolution level."""
        path = SUMMARIES_DIR / f"all_summaries_r{resolution}.json"
        if not path.exists():
            raise FileNotFoundError(f"No summaries found for resolution={resolution}. Run summarize_all() first.")
        with open(path) as f:
            raw = json.load(f)
        # JSON keys are strings; convert back to int
        return {int(k): v for k, v in raw.items()}
