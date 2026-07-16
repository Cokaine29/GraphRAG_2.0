"""
run_evaluation_v4.py — GraphRAG vs Vanilla RAG evaluation on multi-document corpus.
Run: python run_evaluation_v4.py
"""

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import sys, os, json, time, re, shutil, glob
import httpx, anthropic
sys.path.insert(0, '.')

# ── CONFIG ────────────────────────────────────────────────────
# CLAUDE_API_KEY = 'sk-ant-paste-your-key-here'
CLAUDE_API_KEY = 'sk-ant-YOUR_ANTHROPIC_KEY'


MODEL          = 'claude-haiku-4-5-20251001'
VERSION        = 'v2_multi_doc'

if 'paste-your-key' in CLAUDE_API_KEY:
    raise ValueError('Paste your Claude API key first.')

GRAPH_DIR  = f'outputs/graphs/{VERSION}'
GRAPH_PATH = f'{GRAPH_DIR}/knowledge_graph.gml'
SUMM_DIR   = f'outputs/summaries/{VERSION}'
EMBED_DIR  = f'outputs/embeddings/{VERSION}'
EVAL_DIR   = f'evaluation/{VERSION}'

for d in [SUMM_DIR, EMBED_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(GRAPH_PATH):
    raise FileNotFoundError(f'Graph not found at {GRAPH_PATH}. Run build_book_graph.py first.')

print('='*60)
print('GraphRAG Evaluation v4 — Multi-Document Corpus')
print(f'Version: {VERSION}')
print('='*60)
print(f'Graph   : {GRAPH_PATH}')
print(f'Results : {EVAL_DIR}/')

# ── Claude client ─────────────────────────────────────────────
_client = anthropic.Anthropic(
    api_key=CLAUDE_API_KEY,
    http_client=httpx.Client(verify=False)
)

def call_claude(prompt, temperature=0.0, max_tokens=1024):
    msg = _client.messages.create(
        model=MODEL, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()

def parse_json(text):
    text = text.strip()
    text = re.sub(r'^```[a-z]*\n?', '', text)
    text = re.sub(r'\n?```$', '', text).strip()
    s, e = text.find('{'), text.rfind('}')
    if s != -1 and e != -1:
        try: return json.loads(text[s:e+1])
        except: pass
    return {}

# ── Initialize engines ────────────────────────────────────────
print('\n[1/5] Initializing engines...')

from src.config import cfg
cfg['llm']['generation_provider'] = 'ollama'
cfg['llm']['generation_model']    = 'llama3'
cfg['retrieval']['top_communities'] = 5
cfg['retrieval']['map_top_k']       = 5
cfg['chromadb']['persist_dir']      = EMBED_DIR

from src.llm_client import LLMClient
from src.vector_store import VectorStore
from src.community_detection import CommunityDetector
from src.community_summarizer import CommunitySummarizer
from src.graph_query import GraphQueryEngine
from src.chunker import load_all_chunks
from pathlib import Path

llm_local = LLMClient(purpose='generation')

# Vector store — versioned embeddings folder
vs = VectorStore()
if vs.stats()['total_chunks'] == 0:
    print('  Indexing all chunks into versioned vector store...')
    chunks = load_all_chunks()
    vs.build_index(chunks)
print(f'  Vector store: {vs.stats()["total_chunks"]} chunks')

# Community detection — load versioned graph
# Copy partition files from versioned folder to main folder so load_partitions finds them
for f in glob.glob(f'{GRAPH_DIR}/partition_*.json'):
    dst = 'outputs/graphs/' + os.path.basename(f)
    shutil.copy2(f, dst)

cd = CommunityDetector()
cd.load_graph(path=Path(GRAPH_PATH))
cd.load_partitions()

# Community summaries — check versioned folder first
summ_file = f'{SUMM_DIR}/all_summaries_r1.0.json'
if os.path.exists(summ_file):
    # Copy to main summaries folder so CommunitySummarizer finds it
    shutil.copy2(summ_file, 'outputs/summaries/all_summaries_r1.0.json')

cs = CommunitySummarizer(detector=cd, llm=llm_local)
cs.summarize_all(resolution=1.0)

# Save summaries to versioned folder
main_summ = 'outputs/summaries/all_summaries_r1.0.json'
if os.path.exists(main_summ):
    shutil.copy2(main_summ, summ_file)

graph_engine = GraphQueryEngine(summarizer=cs, llm=llm_local)
members = cd.get_community_members(1.0)

print(f'  Graph      : {cd.graph.number_of_nodes()} nodes, {cd.graph.number_of_edges()} edges')
print(f'  Communities: {len(members)} at resolution 1.0')
print('  ✓ All engines ready')

# ── Generation functions ──────────────────────────────────────
VECTOR_PROMPT = """Answer this question using ONLY the provided document passages.
Be specific and precise. Quote exact numbers, names, or formulas when available.

Passages:
{context}

Question: {query}

Answer:"""

def run_vanilla_rag(query):
    cks     = vs.query(query, top_k=5)
    context = "\n\n---\n\n".join(
        [f"[Passage {i+1} from {c['doc_id']}]\n{c['text']}"
         for i, c in enumerate(cks)])
    return llm_local.generate(
        VECTOR_PROMPT.format(context=context, query=query),
        temperature=0.1)

def run_graphrag(query):
    return graph_engine.query(query)['answer']

# ── Questions ─────────────────────────────────────────────────
print('\n[2/5] Loading questions...')

GLOBAL_QUESTIONS = [
    "What are the main architectural innovations introduced in the Transformer compared to RNNs and LSTMs?",
    "How do word embeddings relate to the attention mechanism in Transformers?",
    "What are the key differences between autoregressive and masked language models?",
    "How has the field progressed from RNNs to Transformers to large language models?",
    "What evaluation metrics are used across NLP tasks and why?",
    "How does pre-training and fine-tuning work in modern NLP systems?",
    "What are the main challenges in training large language models?",
    "How does retrieval-augmented generation address limitations of language models?",
    "What role does the encoder-decoder architecture play across NLP tasks?",
    "How do different attention mechanisms compare in terms of efficiency and effectiveness?",
    "What are the main applications of Transformer-based models in NLP?",
    "How does self-supervised learning enable large language models?",
    "What are the key components shared across modern NLP architectures?",
    "How does machine translation benefit from the Transformer architecture?",
    "What are the computational trade-offs in different sequence modeling approaches?",
]

LOCAL_QUESTIONS = [
    "What BLEU score did the Transformer achieve on WMT 2014 English-German translation?",
    "What is the formula for Scaled Dot-Product Attention?",
    "How many attention heads does the base Transformer model use?",
    "What is Byte-Pair Encoding and how does it work?",
    "What is the difference between BERT-base and BERT-large?",
    "What is perplexity and how is it used to evaluate language models?",
    "What optimizer was used to train the original Transformer?",
    "What is the skip-gram objective in Word2Vec?",
    "How does beam search work in machine translation decoding?",
    "What is the masked language modeling objective used to train BERT?",
]

all_questions = (
    [{'id': f'G{i+1:02d}', 'question': q, 'type': 'GLOBAL'}
     for i, q in enumerate(GLOBAL_QUESTIONS)] +
    [{'id': f'L{i+1:02d}', 'question': q, 'type': 'LOCAL'}
     for i, q in enumerate(LOCAL_QUESTIONS)]
)
print(f'  GLOBAL: {len(GLOBAL_QUESTIONS)} | LOCAL: {len(LOCAL_QUESTIONS)} | Total: {len(all_questions)}')

# ── Run both systems ───────────────────────────────────────────
print('\n[3/5] Running both systems...\n')

answers_path = f'{EVAL_DIR}/answers.json'
all_answers  = json.load(open(answers_path)) if os.path.exists(answers_path) else {}
print(f'  Progress: {len(all_answers)}/{len(all_questions)*2}')

for qi, q_item in enumerate(all_questions):
    qid, query, qtype = q_item['id'], q_item['question'], q_item['type']

    key_v = f'{qid}__vanilla_rag'
    if key_v not in all_answers:
        try:
            all_answers[key_v] = {'id': qid, 'question': query, 'type': qtype,
                                   'system': 'vanilla_rag', 'answer': run_vanilla_rag(query)}
        except Exception as e: print(f'  ⚠ vanilla {qid}: {e}')

    key_g = f'{qid}__graphrag'
    if key_g not in all_answers:
        try:
            all_answers[key_g] = {'id': qid, 'question': query, 'type': qtype,
                                   'system': 'graphrag', 'answer': run_graphrag(query)}
        except Exception as e: print(f'  ⚠ graphrag {qid}: {e}')

    with open(answers_path, 'w') as f: json.dump(all_answers, f, indent=2)
    print(f'  [{qtype}] {qid} ({qi+1}/{len(all_questions)}) — {len(all_answers)}/{len(all_questions)*2}')

print(f'\n  ✓ {len(all_answers)} answers collected')

# ── LLM-as-judge ─────────────────────────────────────────────
print('\n[4/5] LLM-as-judge (Claude Haiku)...\n')

JUDGE_PROMPT = """You are evaluating two AI responses about NLP and AI research.

Question: {question}

Response A:
{answer_a}

Response B:
{answer_b}

For each criterion write only A, B, or TIE:
Comprehensiveness - which covers more relevant aspects?
Diversity - which provides more varied insights?
Directness - which more clearly answers the question?

Return ONLY JSON:
{{"comprehensiveness": "A" or "B" or "TIE", "diversity": "A" or "B" or "TIE", "directness": "A" or "B" or "TIE", "reasoning": "one sentence"}}"""

JUDGE_REPEATS  = 3
judgments_path = f'{EVAL_DIR}/judgments.json'
all_judgments  = json.load(open(judgments_path)) if os.path.exists(judgments_path) else {}
total_j        = len(all_questions) * JUDGE_REPEATS
success, failed = 0, 0

print(f'  Total needed: {total_j} | Done: {len(all_judgments)}')

for qi, q_item in enumerate(all_questions):
    qid, query, qtype = q_item['id'], q_item['question'], q_item['type']
    key_v, key_g = f'{qid}__vanilla_rag', f'{qid}__graphrag'
    if key_v not in all_answers or key_g not in all_answers: continue

    ans_v = all_answers[key_v]['answer'][:1000]
    ans_g = all_answers[key_g]['answer'][:1000]

    for repeat in range(JUDGE_REPEATS):
        j_key = f'{qid}__r{repeat}'
        if j_key in all_judgments: success += 1; continue
        try:
            if repeat % 2 == 0: ans_a, ans_b = ans_v, ans_g
            else:                ans_a, ans_b = ans_g, ans_v

            raw     = call_claude(JUDGE_PROMPT.format(question=query, answer_a=ans_a, answer_b=ans_b), temperature=0.1)
            verdict = parse_json(raw)
            if not verdict: failed += 1; continue

            comp = verdict.get('comprehensiveness','TIE')
            div  = verdict.get('diversity','TIE')
            dire = verdict.get('directness','TIE')

            if repeat % 2 != 0:
                swap = {'A':'B','B':'A','TIE':'TIE'}
                comp, div, dire = swap.get(comp,'TIE'), swap.get(div,'TIE'), swap.get(dire,'TIE')

            all_judgments[j_key] = {'qid': qid, 'qtype': qtype, 'repeat': repeat,
                                     'comprehensiveness': comp, 'diversity': div,
                                     'directness': dire, 'reasoning': verdict.get('reasoning','')}
            success += 1
            time.sleep(0.3)
        except Exception as e:
            print(f'  ⚠ {j_key}: {e}'); failed += 1

    with open(judgments_path, 'w') as f: json.dump(all_judgments, f, indent=2)
    if (qi+1) % 5 == 0:
        print(f'  q{qi+1:2d}/{len(all_questions)} — {success} OK, {failed} failed')

print(f'\n  ✓ {success} judgments | {failed} failed')

# ── Compute results ────────────────────────────────────────────
print('\n[5/5] Computing results...')

def compute_win_rates(judgments, qtype_filter=None):
    metrics = ['comprehensiveness','diversity','directness']
    counts  = {m: {'vanilla':0,'graphrag':0,'tie':0} for m in metrics}
    for j in judgments.values():
        if qtype_filter and j.get('qtype') != qtype_filter: continue
        for m in metrics:
            v = j.get(m,'TIE')
            if v=='A': counts[m]['vanilla'] += 1
            elif v=='B': counts[m]['graphrag'] += 1
            else: counts[m]['tie'] += 1
    rates = {}
    for m in metrics:
        total = sum(counts[m].values())
        rates[m] = {'vanilla_rag%': 0,'graphrag%': 0,'tie%': 0,'total': 0} if total==0 else {
            'vanilla_rag%': round(counts[m]['vanilla']/total*100,1),
            'graphrag%':    round(counts[m]['graphrag']/total*100,1),
            'tie%':         round(counts[m]['tie']/total*100,1),
            'total':        total}
    return rates

wr_global = compute_win_rates(all_judgments, 'GLOBAL')
wr_local  = compute_win_rates(all_judgments, 'LOCAL')
wr_all    = compute_win_rates(all_judgments)

json.dump({
    'version': VERSION, 'global_win_rates': wr_global,
    'local_win_rates': wr_local, 'overall_win_rates': wr_all,
    'total_questions': len(all_questions), 'global_questions': len(GLOBAL_QUESTIONS),
    'local_questions': len(LOCAL_QUESTIONS), 'total_judgments': len(all_judgments),
    'successful': success, 'judge_model': MODEL,
    'graph_nodes': cd.graph.number_of_nodes(), 'graph_edges': cd.graph.number_of_edges(),
    'n_communities': len(members),
}, open(f'{EVAL_DIR}/final_results.json','w'), indent=2)

print('\n' + '='*65)
print('FINAL RESULTS — paste into report')
print('='*65)
print(f'\nCorpus: {VERSION} | Graph: {cd.graph.number_of_nodes()} nodes | Communities: {len(members)} | Judge repeats: {JUDGE_REPEATS}x | {success}/{total_j} successful\n')

print('Table 1: GLOBAL queries (15) — GraphRAG expected to win')
print(f'  {"Metric":20s}  {"VanillaRAG%":>12}  {"GraphRAG%":>10}  {"Tie%":>8}')
print('  ' + '-'*55)
for m, r in wr_global.items():
    w = '← GraphRAG' if r['graphrag%'] > r['vanilla_rag%'] else '← Vanilla '
    print(f'  {m:20s}  {r["vanilla_rag%"]:>12.1f}  {r["graphrag%"]:>10.1f}  {r["tie%"]:>8.1f}  {w}')

print(f'\nTable 2: LOCAL queries (10) — Vanilla RAG expected to win')
print(f'  {"Metric":20s}  {"VanillaRAG%":>12}  {"GraphRAG%":>10}  {"Tie%":>8}')
print('  ' + '-'*55)
for m, r in wr_local.items():
    w = '← Vanilla ' if r['vanilla_rag%'] > r['graphrag%'] else '← GraphRAG'
    print(f'  {m:20s}  {r["vanilla_rag%"]:>12.1f}  {r["graphrag%"]:>10.1f}  {r["tie%"]:>8.1f}  {w}')

print(f'\nTable 3: Overall (all 25 questions)')
print(f'  {"Metric":20s}  {"VanillaRAG%":>12}  {"GraphRAG%":>10}  {"Tie%":>8}')
print('  ' + '-'*55)
for m, r in wr_all.items():
    print(f'  {m:20s}  {r["vanilla_rag%"]:>12.1f}  {r["graphrag%"]:>10.1f}  {r["tie%"]:>8.1f}')

print(f'\nResults -> {EVAL_DIR}/final_results.json')
print('✅ Evaluation v4 complete')
