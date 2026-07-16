"""
build_graph_final.py — Knowledge graph from Attention paper using Claude API.
Run: python build_graph_final.py
"""

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import sys, os, json, hashlib, time, glob, re
import httpx, anthropic, fitz, networkx as nx
from collections import defaultdict, Counter
sys.path.insert(0, '.')

# ── CONFIG — paste your key here ──────────────────────────────
CLAUDE_API_KEY = 'sk-ant-YOUR_ANTHROPIC_KEY'
MODEL          = 'claude-haiku-4-5-20251001'
GLEAN_ROUNDS   = 2

if 'paste-your-key' in CLAUDE_API_KEY:
    raise ValueError('Paste your Claude API key first.')

for d in ['data/raw','data/processed','outputs/graphs','cache/extractions']:
    os.makedirs(d, exist_ok=True)

# ── Claude client ──────────────────────────────────────────────
_client = anthropic.Anthropic(
    api_key=CLAUDE_API_KEY,
    http_client=httpx.Client(verify=False)
)

def claude(prompt, max_tokens=1500):
    """Call Claude and return raw text."""
    msg = _client.messages.create(
        model=MODEL, max_tokens=max_tokens, temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()

# ── JSON parser ────────────────────────────────────────────────
def parse(text):
    """Extract JSON object from Claude response regardless of formatting."""
    if not text:
        return {}
    # Remove code fences
    text = re.sub(r'^```[a-z]*\n?', '', text.strip())
    text = re.sub(r'\n?```$', '', text.strip())
    text = text.strip()
    # Find outermost { }
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

print('='*55)
print('GraphRAG — Knowledge Graph Builder (Claude Haiku)')
print('='*55)
print(f'Model: {MODEL} | Gleaning rounds: {GLEAN_ROUNDS}')

# Test
print('\nTesting connection...', end=' ')
r = claude('Reply with exactly: OK', max_tokens=5)
print(r)

# ── Find PDF ───────────────────────────────────────────────────
print('\n[1/5] Finding PDF...')
pdfs = glob.glob('data/raw/*.pdf') + glob.glob('*.pdf')
pdf  = next((p for p in pdfs if 'attention' in p.lower()), None)
if not pdf:
    pdf = input('Path to Attention paper PDF: ').strip().strip('"')
assert os.path.exists(pdf), f'Not found: {pdf}'
print(f'  {pdf}')

# ── Extract + chunk ────────────────────────────────────────────
print('\n[2/5] Extracting text and chunking...')
doc   = fitz.open(pdf)
pages = ['\n'.join(l for l in pg.get_text().split('\n') if len(l.strip())>20)
         for pg in doc]
text  = '\n\n'.join(p for p in pages if p.strip())
print(f'  {len(text):,} chars from {len(pages)} pages')

from src.config import cfg
cfg['llm']['generation_provider'] = 'ollama'
from src.chunker import chunk_document

chunks = chunk_document(text, doc_id='attention')
with open('data/processed/attention_chunks.json','w') as f:
    json.dump(chunks, f, indent=2)
print(f'  {len(chunks)} chunks saved')

# ── Extraction prompts ─────────────────────────────────────────
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

# ── Extract all chunks ─────────────────────────────────────────
print(f'\n[3/5] Extracting (Claude + {GLEAN_ROUNDS} gleaning rounds)...\n')

all_results = []
n_failed    = 0

TRIVIAL = {'this','method','approach','model','system','paper','work','result',
           'output','input','value','function','layer','step','way','type',
           'set','use','section','figure','table','example','problem','task',
           'data','process','note','case','number','size','time','end','base'}

def is_trivial(name):
    return not name or len(name.strip()) < 3 or name.lower().strip() in TRIVIAL

for i, chunk in enumerate(chunks):
    ckey  = 'final_' + hashlib.md5(chunk['text'].encode()).hexdigest()[:10]
    cpath = f'cache/extractions/{ckey}.json'

    if os.path.exists(cpath):
        with open(cpath) as f:
            saved = json.load(f)
        all_results.append(saved)
        print(f'  {i+1:2d}/{len(chunks)} cached — {len(saved["entities"])} ent')
        continue

    entities, rels, claims = [], [], []

    try:
        # Initial extraction
        raw  = claude(EXT_PROMPT.format(text=chunk['text'][:3000]))
        data = parse(raw)
        entities = [e for e in data.get('entities',[]) if isinstance(e,dict) and not is_trivial(e.get('name',''))]
        rels     = [r for r in data.get('relationships',[]) if isinstance(r,dict) and r.get('source') and r.get('target')]
        claims   = [c for c in data.get('claims',[]) if isinstance(c,dict)]

        # Gleaning rounds
        for _ in range(GLEAN_ROUNDS):
            existing = ', '.join(e['name'] for e in entities[:10])
            graw  = claude(GLEAN_PROMPT.format(entities=existing, text=chunk['text'][:2000]))
            gdata = parse(graw)
            new_e = [e for e in gdata.get('entities',[])      if isinstance(e,dict) and not is_trivial(e.get('name',''))]
            new_r = [r for r in gdata.get('relationships',[]) if isinstance(r,dict) and r.get('source') and r.get('target')]
            new_c = [c for c in gdata.get('claims',[])        if isinstance(c,dict)]
            entities += new_e
            rels     += new_r
            claims   += new_c
            time.sleep(0.3)

    except Exception as ex:
        print(f'  {i+1:2d}/{len(chunks)} FAILED: {ex}')
        n_failed += 1

    saved = {'chunk_id': chunk['chunk_id'], 'entities': entities, 'rels': rels, 'claims': claims}
    with open(cpath,'w') as f: json.dump(saved, f, indent=2)
    all_results.append(saved)
    print(f'  {i+1:2d}/{len(chunks)} — {len(entities)} ent, {len(rels)} rel, {len(claims)} claims')
    time.sleep(0.5)

total_e = sum(len(r['entities']) for r in all_results)
total_r = sum(len(r['rels'])     for r in all_results)
total_c = sum(len(r['claims'])   for r in all_results)
print(f'\n  Total: {total_e} entities, {total_r} relations, {total_c} claims | Failed: {n_failed}')

# ── Entity resolution ──────────────────────────────────────────
print('\n[4/5] Entity resolution and graph construction...')

def norm(name):
    n = name.lower().strip().replace('-',' ').replace('_',' ')
    for p in ['the ','a ','an ']: n = n[len(p):] if n.startswith(p) else n
    return n.strip()

raw_ent = defaultdict(list)
for r in all_results:
    for e in r['entities']:
        name = e.get('name','').strip()
        if name and not is_trivial(name):
            raw_ent[norm(name)].append({
                'raw': name, 'type': e.get('type','concept'),
                'desc': e.get('description',''), 'chunk': r['chunk_id']
            })

cmap, edata = {}, {}
for nname, occs in raw_ent.items():
    canon = Counter(o['raw'] for o in occs).most_common(1)[0][0]
    descs = list(set(o['desc'] for o in occs if o['desc']))
    etype = Counter(o['type'] for o in occs).most_common(1)[0][0]
    edata[canon] = {
        'entity_type':   etype,
        'description':   ' '.join(descs[:2])[:300],
        'frequency':     len(occs),
        'source_chunks': ','.join(set(o['chunk'] for o in occs))[:150],
        'claims':        '',
    }
    for o in occs: cmap[o['raw']] = canon

# Attach claims to nodes
for r in all_results:
    for c in r['claims']:
        subj = c.get('subject','').strip()
        canon = cmap.get(subj, subj)
        if canon in edata:
            existing = edata[canon]['claims']
            new_claim = c.get('claim','')
            if new_claim and new_claim not in existing:
                edata[canon]['claims'] = (existing + ' | ' + new_claim if existing else new_claim)[:300]

print(f'  Unique entities: {len(edata)}')

# Claude merge
MERGE = '''\
These entity names came from "Attention is All You Need". Which ones mean the same thing?

{names}

Return ONLY a JSON array. Each inner array = merge group, first item = name to keep.
Example: [["Self-Attention","self-attention mechanism","Self Attention Layer"]]
If no merges: []'''

merge_map = {}
try:
    mraw = claude(MERGE.format(names='\n'.join(f'- {n}' for n in list(edata.keys())[:50])))
    mraw = re.sub(r'^```[a-z]*\n?','',mraw.strip())
    mraw = re.sub(r'\n?```$','',mraw.strip())
    s, e = mraw.find('['), mraw.rfind(']')
    if s != -1 and e != -1:
        groups = json.loads(mraw[s:e+1])
        n_merged = 0
        for g in groups:
            if len(g) < 2: continue
            keep = g[0]
            for dup in g[1:]:
                if dup in edata and keep in edata and dup != keep:
                    edata[keep]['frequency'] += edata[dup].get('frequency',1)
                    merge_map[dup] = keep
                    n_merged += 1
        for dup in merge_map: edata.pop(dup, None)
        print(f'  Merged {n_merged} duplicates → {len(edata)} final entities')
except Exception as ex:
    print(f'  Merge skipped: {ex}')

# ── Build graph ────────────────────────────────────────────────
print('\n[5/5] Building graph...')

G = nx.MultiDiGraph()
for name, d in edata.items():
    G.add_node(name, **d)

skipped = 0
for r in all_results:
    for rel in r['rels']:
        src_raw = (rel.get('source') or '').strip()
        tgt_raw = (rel.get('target') or '').strip()
        src = merge_map.get(cmap.get(src_raw, src_raw), cmap.get(src_raw, src_raw))
        tgt = merge_map.get(cmap.get(tgt_raw, tgt_raw), cmap.get(tgt_raw, tgt_raw))
        if src not in G or tgt not in G or src == tgt:
            skipped += 1; continue
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
        ereg[k]['count']  += 1

GC = nx.DiGraph()
for n, d in G.nodes(data=True): GC.add_node(n, **d)
for (s,t,rel), ed in ereg.items():
    GC.add_edge(s, t, relation_type=rel, description=ed['desc'],
                weight=ed['weight'], computed_weight=ed['weight'], edge_count=ed['count'])

iso = [n for n in GC if GC.degree(n)==0]
GC.remove_nodes_from(iso)

print(f'  Nodes: {GC.number_of_nodes()} | Edges: {GC.number_of_edges()} | Isolated removed: {len(iso)}')

print('\n  Top entities:')
for name, deg in sorted(GC.degree(), key=lambda x: x[1], reverse=True)[:10]:
    print(f'    {deg:3d}  {name}')

nx.write_gml(GC, 'outputs/graphs/knowledge_graph.gml')
json.dump({'nodes':GC.number_of_nodes(),'edges':GC.number_of_edges(),
           'model':MODEL,'glean_rounds':GLEAN_ROUNDS},
          open('outputs/graphs/graph_stats.json','w'), indent=2)

print(f'\n  Saved → outputs/graphs/knowledge_graph.gml')
print('\n' + '='*55)
print(f'DONE: {GC.number_of_nodes()} nodes, {GC.number_of_edges()} edges')
print('='*55)
print('''
Next steps:
  Remove-Item -Force outputs\\graphs\\partition_*.json
  Remove-Item -Recurse -Force outputs\\summaries
  New-Item -ItemType Directory -Force outputs\\summaries
  python -c "import sys;sys.path.insert(0,'.');from src.config import cfg;cfg['llm']['generation_provider']='ollama';cfg['llm']['generation_model']='llama3';from src.community_detection import CommunityDetector;cd=CommunityDetector();cd.load_graph();cd.detect_all_levels();print(cd.stats())"
  python run_evaluation_v3.py
''')
