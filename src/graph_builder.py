"""
graph_builder.py - Builds a knowledge graph from document chunks using LLM extraction.
"""

import json
import re
import os
import hashlib
import time
import logging
from typing import List, Dict, Any, Optional
import networkx as nx

from src.llm_client import LLMClient
from src.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)

__all__ = ["GraphBuilder"]

EXT_PROMPT = '''\
Extract a knowledge graph from this research paper chunk.

TEXT:
{text}

Return a JSON object with exactly these three keys:
{{
  "entities": [
    {{"name": "Transformer", "type": "model", "description": "encoder-decoder architecture"}}
  ],
  "relationships": [
    {{"source": "Transformer", "relation": "uses", "target": "Self-Attention", "weight": 9}}
  ],
  "claims": [
    {{"subject": "Transformer", "claim": "achieves 28.4 BLEU on WMT EN-DE"}}
  ]
}}

Rules for entities:
- Use canonical names: Transformer, Self-Attention, Multi-Head Attention, BLEU Score, Adam
- Skip generic words: model, method, approach, system, result, layer, step, work, paper
- Type must be one of: model, algorithm, technique, concept, dataset, metric, task
- Extract 4-8 entities

Rules for relationships:
- source and target must be entity names you extracted
- relation must be one of: uses, based_on, extends, improves_on, part_of, enables, introduces, compared_to, trained_with, evaluated_on
- weight is 1-10

Rules for claims:
- Must be specific facts with numbers or comparisons
- subject must be an entity name you extracted'''

GLEAN_PROMPT = '''\
You already extracted these entities from a research paper chunk:
{entities}

Here is the original text again:
{text}

What did you MISS? Extract additional entities, relationships, and claims that are NOT in the list above.
Focus on: numbers, dataset names, training details, evaluation metrics.

Return JSON with same format:
{{
  "entities": [...],
  "relationships": [...],
  "claims": [...]
}}

If nothing new: {{"entities":[],"relationships":[],"claims":[]}}'''


class GraphBuilder:
    """Builds a knowledge graph from document chunks using LLM extraction."""
    
    def __init__(self, llm: Optional[LLMClient] = None, resolver: Optional[EntityResolver] = None):
        self.llm = llm or LLMClient(purpose="extraction")
        self.resolver = resolver or EntityResolver(llm=self.llm)
        self.glean_rounds = 2
        self.cache_dir = "cache/extractions"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _parse_json(self, text: str) -> Dict:
        if not text:
            return {}
        text = re.sub(r'^```[a-z]*\n?', '', text.strip())
        text = re.sub(r'\n?```$', '', text.strip())
        text = text.strip()
        depth, start, result = 0, -1, None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0: start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        result = json.loads(text[start:i+1])
                        break
                    except:
                        pass
        return result or {}

    def extract_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities, relationships, claims from a single chunk."""
        ckey = 'final_' + hashlib.md5(chunk['text'].encode()).hexdigest()[:10]
        cpath = f"{self.cache_dir}/{ckey}.json"
        
        if os.path.exists(cpath):
            with open(cpath) as f:
                saved = json.load(f)
            logger.info(f"Loaded chunk from cache: {cpath}")
            return saved

        entities, rels, claims = [], [], []
        try:
            raw = self.llm.generate(EXT_PROMPT.format(text=chunk['text'][:1000]), temperature=0.0)
            data = self._parse_json(raw)
            entities = [e for e in data.get('entities',[]) if isinstance(e,dict) and not self.resolver.is_trivial(e.get('name',''))]
            rels = [r for r in data.get('relationships',[]) if isinstance(r,dict) and r.get('source') and r.get('target')]
            claims = [c for c in data.get('claims',[]) if isinstance(c,dict)]

            # Gleaning rounds
            for _ in range(self.glean_rounds):
                if not entities:
                    break
                existing = ', '.join(e.get('name','') for e in entities[:10])
                graw = self.llm.generate(GLEAN_PROMPT.format(entities=existing, text=chunk['text'][:800]), temperature=0.0)
                gdata = self._parse_json(graw)
                new_e = [e for e in gdata.get('entities',[]) if isinstance(e,dict) and not self.resolver.is_trivial(e.get('name',''))]
                new_r = [r for r in gdata.get('relationships',[]) if isinstance(r,dict) and r.get('source') and r.get('target')]
                new_c = [c for c in gdata.get('claims',[]) if isinstance(c,dict)]
                entities += new_e
                rels += new_r
                claims += new_c
                time.sleep(0.3)

        except Exception as ex:
            logger.error(f"Extraction failed for chunk {chunk.get('chunk_id')}: {ex}")

        saved = {'chunk_id': chunk['chunk_id'], 'entities': entities, 'rels': rels, 'claims': claims}
        with open(cpath,'w') as f:
            json.dump(saved, f, indent=2)
        return saved

    def build(self, chunks: List[Dict[str, Any]]) -> nx.DiGraph:
        """Full pipeline: extract -> resolve -> construct graph."""
        logger.info(f"Extracting knowledge from {len(chunks)} chunks...")
        all_results = []
        for chunk in chunks:
            res = self.extract_chunk(chunk)
            all_results.append(res)

        logger.info("Resolving entities...")
        edata, cmap = self.resolver.resolve_entities(all_results)
        merge_map = self.resolver.merge_duplicates_llm(edata, cmap)

        logger.info("Building graph...")
        G = nx.MultiDiGraph()
        for name, d in edata.items():
            G.add_node(name, **d)

        skipped = 0
        for r in all_results:
            for rel in r.get('rels', []):
                src_raw = (rel.get('source') or '').strip()
                tgt_raw = (rel.get('target') or '').strip()
                src = merge_map.get(cmap.get(src_raw, src_raw), cmap.get(src_raw, src_raw))
                tgt = merge_map.get(cmap.get(tgt_raw, tgt_raw), cmap.get(tgt_raw, tgt_raw))
                
                if src not in G or tgt not in G or src == tgt:
                    skipped += 1
                    continue
                
                G.add_edge(src, tgt,
                           relation_type=rel.get('relation','related_to'),
                           description=(rel.get('description') or ''),
                           weight=float(rel.get('weight') or 5),
                           computed_weight=float(rel.get('weight') or 5))

        # Consolidate parallel edges
        ereg = {}
        for s, t, d in G.edges(data=True):
            k = (s, t, d.get('relation_type','related_to'))
            if k not in ereg:
                ereg[k] = {'weight': d['weight'], 'desc': d.get('description',''), 'count': 1}
            else:
                ereg[k]['weight'] += d['weight']
                ereg[k]['count'] += 1

        GC = nx.DiGraph()
        for n, d in G.nodes(data=True):
            GC.add_node(n, **d)
        for (s, t, rel), ed in ereg.items():
            GC.add_edge(s, t, relation_type=rel, description=ed['desc'],
                        weight=ed['weight'], computed_weight=ed['weight'], edge_count=ed['count'])

        iso = [n for n in GC if GC.degree(n) == 0]
        GC.remove_nodes_from(iso)
        logger.info(f"Graph built with {GC.number_of_nodes()} nodes and {GC.number_of_edges()} edges. Removed {len(iso)} isolated nodes.")
        return GC
