"""
entity_resolver.py - resolves and merges entities extracted from chunks.
"""

import json
import re
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

__all__ = ["EntityResolver"]

MERGE_PROMPT = """These entity names came from a research corpus. Which ones mean the same thing?

{names}

Return ONLY a JSON array. Each inner array = merge group, first item = name to keep.
Example: [["Self-Attention","self-attention mechanism","Self Attention Layer"]]
If no merges: []"""


class EntityResolver:
    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient(purpose="generation")
        self.trivial_words = {'this','method','approach','model','system','paper','work','result',
                              'output','input','value','function','layer','step','way','type',
                              'set','use','section','figure','table','example','problem','task',
                              'data','process','note','case','number','size','time','end','base'}

    def is_trivial(self, name: str) -> bool:
        return not name or len(name.strip()) < 3 or name.lower().strip() in self.trivial_words

    def norm(self, name: str) -> str:
        n = name.lower().strip().replace('-',' ').replace('_',' ')
        for p in ['the ','a ','an ']:
            n = n[len(p):] if n.startswith(p) else n
        return n.strip()

    def resolve_entities(self, all_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Process raw chunk extractions, canonicalize names, and consolidate claims.
        Returns (entity_data, cmap) where:
        - entity_data: dict of canonical_name -> attributes
        - cmap: dict of raw_name -> canonical_name
        """
        raw_ent = defaultdict(list)
        for r in all_results:
            for e in r.get('entities', []):
                name = e.get('name', '').strip()
                if name and not self.is_trivial(name):
                    raw_ent[self.norm(name)].append({
                        'raw': name,
                        'type': e.get('type', 'concept'),
                        'desc': e.get('description', ''),
                        'chunk': r.get('chunk_id')
                    })

        cmap: Dict[str, str] = {}
        edata: Dict[str, Any] = {}
        
        for nname, occs in raw_ent.items():
            canon = Counter(o['raw'] for o in occs).most_common(1)[0][0]
            descs = list(set(o['desc'] for o in occs if o['desc']))
            etype = Counter(o['type'] for o in occs).most_common(1)[0][0]
            edata[canon] = {
                'entity_type':   etype,
                'description':   ' '.join(descs[:2])[:300],
                'frequency':     len(occs),
                'source_chunks': ','.join(set(o['chunk'] for o in occs if o['chunk']))[:150],
                'claims':        '',
            }
            for o in occs:
                cmap[o['raw']] = canon

        # Attach claims to nodes
        for r in all_results:
            for c in r.get('claims', []):
                subj = c.get('subject', '').strip()
                canon = cmap.get(subj, subj)
                if canon in edata:
                    existing = edata[canon]['claims']
                    new_claim = c.get('claim', '')
                    if new_claim and new_claim not in existing:
                        edata[canon]['claims'] = (existing + ' | ' + new_claim if existing else new_claim)[:300]

        logger.info(f"Unique entities before LLM merge: {len(edata)}")
        return edata, cmap

    def merge_duplicates_llm(self, edata: Dict[str, Any], cmap: Dict[str, str]) -> Dict[str, str]:
        """
        Use LLM to find aliases among the canonical entities.
        Updates edata in place and returns merge_map (dup_name -> kept_name).
        """
        merge_map: Dict[str, str] = {}
        names_to_merge = list(edata.keys())[:50]  # Take top 50 for cost/context reasons
        if not names_to_merge:
            return merge_map

        try:
            prompt = MERGE_PROMPT.format(names='\n'.join(f'- {n}' for n in names_to_merge))
            mraw = self.llm.generate(prompt, temperature=0.0)
            mraw = re.sub(r'^```[a-z]*\n?', '', mraw.strip())
            mraw = re.sub(r'\n?```$', '', mraw.strip())
            s, e = mraw.find('['), mraw.rfind(']')
            if s != -1 and e != -1:
                groups = json.loads(mraw[s:e+1])
                n_merged = 0
                for g in groups:
                    if len(g) < 2: continue
                    keep = g[0]
                    for dup in g[1:]:
                        if dup in edata and keep in edata and dup != keep:
                            edata[keep]['frequency'] += edata[dup].get('frequency', 1)
                            merge_map[dup] = keep
                            n_merged += 1
                
                # Remove duplicates from edata
                for dup in merge_map:
                    edata.pop(dup, None)
                
                logger.info(f"LLM merged {n_merged} duplicates -> {len(edata)} final entities")
        except Exception as ex:
            logger.warning(f"LLM Merge skipped or failed: {ex}")

        return merge_map

