"""
evaluation.py - Runs GraphRAG vs Vanilla RAG evaluation with LLM-as-judge.
"""

import json
import re
import time
import logging
from typing import List, Dict, Any, Optional

from src.llm_client import LLMClient
from src.vector_store import VectorStore
from src.graph_query import GraphQueryEngine

logger = logging.getLogger(__name__)

__all__ = ["Evaluator"]

JUDGE_PROMPT = """You are evaluating two AI responses about IoT, Blockchain, and supply chain management.

Question: {question}

Response A:
{answer_a}

Response B:
{answer_b}

For each criterion write only A, B, or TIE:
Comprehensiveness - which covers more relevant aspects?
Diversity - which provides more varied insights?
Directness - which more clearly answers the question?
Faithfulness - which answer relies strictly on information provided without hallucinating?

Return ONLY JSON:
{{"comprehensiveness": "A" or "B" or "TIE", "diversity": "A" or "B" or "TIE", "directness": "A" or "B" or "TIE", "faithfulness": "A" or "B" or "TIE", "reasoning": "one sentence"}}"""

VECTOR_PROMPT = """Answer this question using ONLY the provided document passages.
Be specific and precise. Quote exact numbers, names, or formulas when available.

Passages:
{context}

Question: {query}

Answer:"""


class Evaluator:
    """Runs GraphRAG vs Vanilla RAG evaluation with LLM-as-judge."""
    
    def __init__(self, judge_llm: Optional[LLMClient] = None, generator_llm: Optional[LLMClient] = None,
                 vector_store: Optional[VectorStore] = None, graph_engine: Optional[GraphQueryEngine] = None):
        if not judge_llm:
            judge_llm = LLMClient(purpose="generation")
            judge_llm.model = "llama-3.3-70b-versatile"
        self.judge_llm = judge_llm
        self.generator_llm = generator_llm or LLMClient(purpose="generation")
        self.vector_store = vector_store or VectorStore()
        self.graph_engine = graph_engine
        self.judge_repeats = 3

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        text = re.sub(r'^```[a-z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text).strip()
        s, e = text.find('{'), text.rfind('}')
        if s != -1 and e != -1:
            try: return json.loads(text[s:e+1])
            except: pass
        return {}

    def _run_vanilla_rag(self, query: str) -> str:
        cks = self.vector_store.query(query, top_k=5)
        context = "\n\n---\n\n".join([f"[Passage {i+1} from {c['doc_id']}]\n{c['text']}" for i, c in enumerate(cks)])
        return self.generator_llm.generate(VECTOR_PROMPT.format(context=context, query=query), temperature=0.1)

    def _run_graphrag(self, query: str) -> str:
        if not self.graph_engine:
            return "Graph engine not initialized."
        return self.graph_engine.query(query)['answer']

    def run(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run full evaluation pipeline."""
        logger.info(f"Running evaluation on {len(questions)} questions...")
        
        all_answers = {}
        # Collect answers
        for qi, q_item in enumerate(questions):
            qid, query, qtype = q_item.get('id', f'q{qi}'), q_item.get('question', ''), q_item.get('type', 'LOCAL')
            
            # Vanilla RAG
            key_v = f"{qid}__vanilla_rag"
            try:
                ans = self._run_vanilla_rag(query)
                all_answers[key_v] = {'id': qid, 'question': query, 'type': qtype, 'system': 'vanilla_rag', 'answer': ans}
            except Exception as e:
                logger.error(f"Vanilla RAG failed for {qid}: {e}")

            # GraphRAG
            key_g = f"{qid}__graphrag"
            try:
                ans = self._run_graphrag(query)
                all_answers[key_g] = {'id': qid, 'question': query, 'type': qtype, 'system': 'graphrag', 'answer': ans}
            except Exception as e:
                logger.error(f"GraphRAG failed for {qid}: {e}")
                
        # Run judge
        logger.info("Running LLM-as-judge...")
        all_judgments = {}
        success, failed = 0, 0
        
        for qi, q_item in enumerate(questions):
            qid, query, qtype = q_item.get('id', f'q{qi}'), q_item.get('question', ''), q_item.get('type', 'LOCAL')
            key_v, key_g = f'{qid}__vanilla_rag', f'{qid}__graphrag'
            if key_v not in all_answers or key_g not in all_answers:
                continue

            ans_v = all_answers[key_v]['answer'][:1000]
            ans_g = all_answers[key_g]['answer'][:1000]

            for repeat in range(self.judge_repeats):
                j_key = f'{qid}__r{repeat}'
                try:
                    if repeat % 2 == 0:
                        ans_a, ans_b = ans_v, ans_g
                    else:
                        ans_a, ans_b = ans_g, ans_v

                    raw = self.judge_llm.generate(JUDGE_PROMPT.format(question=query, answer_a=ans_a, answer_b=ans_b), temperature=0.1)
                    verdict = self._parse_json(raw)
                    if not verdict:
                        failed += 1
                        continue

                    comp = verdict.get('comprehensiveness', 'TIE')
                    div = verdict.get('diversity', 'TIE')
                    dire = verdict.get('directness', 'TIE')
                    faith = verdict.get('faithfulness', 'TIE')

                    if repeat % 2 != 0:
                        swap = {'A': 'B', 'B': 'A', 'TIE': 'TIE'}
                        comp, div, dire, faith = swap.get(comp, 'TIE'), swap.get(div, 'TIE'), swap.get(dire, 'TIE'), swap.get(faith, 'TIE')

                    all_judgments[j_key] = {
                        'qid': qid, 'qtype': qtype, 'repeat': repeat,
                        'comprehensiveness': comp, 'diversity': div, 'directness': dire, 'faithfulness': faith,
                        'reasoning': verdict.get('reasoning', '')
                    }
                    success += 1
                    time.sleep(0.3)
                except Exception as e:
                    logger.error(f"Judge failed for {j_key}: {e}")
                    failed += 1

        logger.info(f"Judgments complete: {success} successful, {failed} failed")
        
        return {
            "answers": all_answers,
            "judgments": all_judgments,
            "win_rates": self.compute_win_rates(all_judgments)
        }

    def compute_win_rates(self, judgments: Dict[str, Any], qtype_filter: Optional[str] = None) -> Dict[str, Any]:
        metrics = ['comprehensiveness', 'diversity', 'directness', 'faithfulness']
        counts = {m: {'vanilla': 0, 'graphrag': 0, 'tie': 0} for m in metrics}
        for j in judgments.values():
            if qtype_filter and j.get('qtype') != qtype_filter:
                continue
            for m in metrics:
                v = j.get(m, 'TIE')
                if v == 'A': counts[m]['vanilla'] += 1
                elif v == 'B': counts[m]['graphrag'] += 1
                else: counts[m]['tie'] += 1
                
        rates = {}
        for m in metrics:
            total = sum(counts[m].values())
            rates[m] = {'vanilla_rag%': 0, 'graphrag%': 0, 'tie%': 0, 'total': 0} if total == 0 else {
                'vanilla_rag%': round(counts[m]['vanilla'] / total * 100, 1),
                'graphrag%': round(counts[m]['graphrag'] / total * 100, 1),
                'tie%': round(counts[m]['tie'] / total * 100, 1),
                'total': total
            }
        return rates
