"""
app_v2.py — GraphRAG Streamlit interface (Premium Version).

Tabs:
  1. Chat       — ask questions, see routing + traversal graph
  2. Ingest     — upload documents
  3. Knowledge Graph — full interactive graph
  4. Communities — browse community summaries
"""

import json, shutil, glob, os
import streamlit as st
import networkx as nx
from pathlib import Path

# ── Point to main graph ───────────────────────────────
import sys
sys.path.insert(0, '.')
from src.config import cfg

st.set_page_config(
    page_title="GraphRAG (Groq Edition)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Premium CSS ─────────────────────────────────────────
st.markdown("""
<style>
    /* Chat window custom scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    
    /* Custom headers */
    h1, h2, h3, h4, h5, h6 { color: #1e293b !important; font-family: 'Inter', sans-serif; }
    
    /* Sleek buttons */
    .stButton>button {
        border-radius: 8px !important;
        border: 1px solid #10b981 !important;
        background-color: #ecfdf5 !important;
        color: #059669 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #10b981 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    /* User message bubble */
    [data-testid='stChatMessage']:nth-child(odd) {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Assistant message bubble */
    [data-testid='stChatMessage']:nth-child(even) {
        background-color: #ffffff;
        border-left: 4px solid #10b981;
        border-radius: 0 12px 12px 0;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Highlight chat input explicitly so it isn't invisible */
    [data-testid="stChatInput"] {
        border: 2px solid #10b981 !important;
        background-color: #ffffff !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Premium App Look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
    }
</style>
""", unsafe_allow_html=True)


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

class StreamlitLogHandler(logging.Handler):
    def __init__(self, status_container):
        super().__init__()
        self.status_container = status_container
    def emit(self, record):
        msg = self.format(record)
        self.status_container.write(f"🔄 {msg}")

# ── Load engines ──────────────────────────────────────────────
@st.cache_resource
def load_engine():
    from src.llm_client import LLMClient
    from src.vector_store import VectorStore
    from src.router import QueryRouter
    from src.query_engine import QueryEngine
    from src.config import GRAPH_PATH

    llm    = LLMClient(provider="groq", model="llama-3.1-8b-instant")
    vs     = VectorStore()
    router = QueryRouter(llm=llm)

    graph_engine    = None
    graph_traversal = None

    if GRAPH_PATH.exists():
        try:
            from src.community_detection import CommunityDetector
            from src.community_summarizer import CommunitySummarizer
            from src.graph_query import GraphQueryEngine
            from src.graph_traversal import GraphTraversal

            cd = CommunityDetector()
            cd.load_graph()
            cd.load_partitions()
            cs           = CommunitySummarizer(detector=cd, llm=llm)
            graph_engine = GraphQueryEngine(summarizer=cs, llm=llm)

            gt = GraphTraversal(llm=llm)
            gt.load_graph()
            graph_traversal = gt
        except Exception as e:
            import logging
            logging.warning(f"Graph not loaded: {e}")

    return QueryEngine(
        vector_store=vs,
        graph_engine=graph_engine,
        graph_traversal=graph_traversal,
        router=router,
        llm=llm,
    )

@st.cache_resource
def load_graph():
    from src.config import GRAPH_PATH
    if GRAPH_PATH.exists():
        return nx.read_gml(str(GRAPH_PATH))
    return None


# ── Traversal visualization ───────────────────────────────────
def render_traversal_graph(G, seed_nodes, subgraph_nodes, physics=True):
    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components
        import tempfile, os

        if not subgraph_nodes:
            return

        subG = G.subgraph(list(subgraph_nodes))
        net  = Network(height="450px", width="100%",
                       bgcolor="#0b0f19", font_color="#ffffff",
                       directed=True)

        for node in subG.nodes():
            if node in seed_nodes:
                color = "#ff2a5f"
                size  = 35
                title = f"🔴 QUERY ENTITY: {node}\n{G.nodes[node].get('description','')[:100]}"
            else:
                color = "#ff9a00"
                size  = 20
                title = f"🟠 {node}\n{G.nodes[node].get('description','')[:100]}"

            net.add_node(node, label=node, color=color, size=size,
                         title=title, font={"size": 14, "face": "Inter", "color": "#fff", "strokeWidth": 2, "strokeColor": "#000"})

        for src, tgt, data in subG.edges(data=True):
            rel = data.get("relation_type", "")
            net.add_edge(src, tgt, label=rel,
                         color="#556677", arrows="to",
                         title=data.get("description",""),
                         font={"size": 10, "face": "Inter", "color": "#64748b", "strokeWidth": 1, "strokeColor": "#0b0f19"})

        physics_enabled = "true" if physics else "false"
        net.set_options(f'''{{
            "physics": {{"enabled": {physics_enabled}, "stabilization": {{"iterations": 150}}}},
            "edges": {{"smooth": {{"type": "continuous"}}}},
            "interaction": {{"hover": true}}
        }}''')

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as f:
            net.save_graph(f.name)
            html_path = f.name

        with open(html_path) as f:
            html = f.read()
        os.unlink(html_path)

        components.html(html, height=470)

    except ImportError:
        st.warning("Install `pyvis` for interactive visualizations.")

ROUTE_COLORS = {
    "LOCAL-VECTOR":    "🔵",
    "LOCAL-TRAVERSAL": "🟣",
    "GLOBAL":          "🟢",
    "HYBRID":          "🟠",
}


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1>🌌 GraphRAG Studio</h1>", unsafe_allow_html=True)
    st.caption("Advanced Graph-Based RAG Dashboard")
    st.divider()

    st.subheader("⚙️ Query Settings")
    force_route = st.selectbox(
        "Routing Strategy",
        ["🤖 Auto (Router decides)", "🔵 LOCAL", "🟣 LOCAL-TRAVERSAL", "🟢 GLOBAL", "🟠 HYBRID"],
    )
    force = None if force_route.startswith("🤖") else force_route.split()[1]

    llm_temp = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1, help="Higher = more creative, Lower = more deterministic")
    top_k = st.slider("Retrieval Top-K", 1, 20, 5, help="Number of chunks to retrieve for Local RAG")

    st.divider()
    st.subheader("📊 System Status")
    try:
        from src.vector_store import VectorStore
        from src.config import GRAPH_PATH
        vs    = VectorStore()
        stats = vs.stats()
        
        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0;; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
            <div style="font-size: 0.8rem; color: #64748b;">Chunks Indexed</div>
            <div style="font-size: 1.5rem; color: #10b981; font-weight: 600;">{stats['total_chunks']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if GRAPH_PATH.exists():
            G_side = nx.read_gml(str(GRAPH_PATH))
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0;; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                <div style="font-size: 0.8rem; color: #64748b;">Graph Nodes</div>
                <div style="font-size: 1.5rem; color: #059669; font-weight: 600;">{G_side.number_of_nodes()}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0;; padding: 10px; border-radius: 8px;">
                <div style="font-size: 0.8rem; color: #64748b;">Graph Edges</div>
                <div style="font-size: 1.5rem; color: #059669; font-weight: 600;">{G_side.number_of_edges()}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Graph missing.")
    except Exception:
        st.info("Not yet indexed")

# End of sidebar

from streamlit_option_menu import option_menu
page = option_menu(
    menu_title=None,
    options=["💬 RAG Chat", "🕸 Graph Explorer", "🗂 Communities", "📄 Document Ingestion", "ℹ️ About Project"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "5px!important", "background-color": "#ffffff", "border": "1px solid #e2e8f0", "border-radius": "10px"},
        "icon": {"display": "none"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#ecfdf5", "color": "#475569"},
        "nav-link-selected": {"background-color": "#10b981", "color": "white", "font-weight": "600"},
    }
)
st.divider()


# ══════════════════════════════════════════════════════════════
# PAGE 1 — CHAT
# ══════════════════════════════════════════════════════════════
if page == "💬 RAG Chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    G_viz = load_graph()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "thinking" in msg and msg["thinking"]:
                with st.expander("💭 Thinking Process"):
                    st.write(msg["thinking"])
            st.write(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta  = msg["meta"]
                route = meta.get("route", "?")
                emoji = ROUTE_COLORS.get(route, "⚪")

                st.markdown(f"""
                <div style="display: flex; gap: 15px; margin-top: 10px; padding: 10px; background: #ffffff; border: 1px solid #e2e8f0;; border-radius: 8px; border-left: 4px solid #10b981;">
                    <div><span style="color:#64748b; font-size: 0.8rem;">Route</span><br><b>{emoji} {route}</b></div>
                    <div><span style="color:#64748b; font-size: 0.8rem;">Confidence</span><br><b>{meta.get('confidence', 0):.0%}</b></div>
                    <div><span style="color:#64748b; font-size: 0.8rem;">Pipeline</span><br><b>{meta.get('pipeline', '?')}</b></div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("🧠 Internal Reasoning"):
                    st.write(meta.get("reasoning", ""))

                if route == "LOCAL-TRAVERSAL" and G_viz and "traversal" in meta:
                    with st.expander("🕸 Traversed Subgraph", expanded=True):
                        render_traversal_graph(
                            G_viz,
                            seed_nodes=set(meta["traversal"].get("seed_entities", [])),
                            subgraph_nodes=set(meta["traversal"].get("subgraph_nodes", [])),
                        )

    if query := st.chat_input("Ask a highly complex question across your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            # Load engine BEFORE setting up UI loggers to prevent Streamlit CacheReplayClosureError
            engine = load_engine()
            
            with st.status("Synthesizing response (this may take a few minutes locally)...", expanded=True) as status_box:
                log_handler = StreamlitLogHandler(status_box)
                log_handler.setFormatter(logging.Formatter('%(message)s'))
                loggers = ['src.router', 'src.graph_query', 'src.community_summarizer', 'src.vector_store', 'src.llm_client']
                for lname in loggers:
                    logging.getLogger(lname).setLevel(logging.INFO)
                    logging.getLogger(lname).addHandler(log_handler)
                
                try:
                    # Apply slider settings
                    engine.llm.temperature = llm_temp
                    engine.vector_store.top_k = top_k
                    
                    result = engine.query(query, force_route=force)
                    status_box.update(label="Response Generated!", state="complete", expanded=True)
                    answer = result["answer"]
                    
                    import re
                    thinking = ""
                    think_match = re.search(r"<think>(.*?)</think>", answer, flags=re.DOTALL)
                    if think_match:
                        thinking = think_match.group(1).strip()
                        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
                        with st.expander("💭 Thinking Process"):
                            st.write(thinking)

                    route  = result["route"]
                    emoji  = ROUTE_COLORS.get(route, "⚪")

                    st.write(answer)

                    st.markdown(f"""
                    <div style="display: flex; gap: 15px; margin-top: 10px; padding: 10px; background: #ffffff; border: 1px solid #e2e8f0;; border-radius: 8px; border-left: 4px solid #10b981;">
                        <div><span style="color:#64748b; font-size: 0.8rem;">Route</span><br><b>{emoji} {route}</b></div>
                        <div><span style="color:#64748b; font-size: 0.8rem;">Confidence</span><br><b>{result['confidence']:.0%}</b></div>
                        <div><span style="color:#64748b; font-size: 0.8rem;">Pipeline</span><br><b>{result["metadata"].get("pipeline","?")}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("🧠 Internal Reasoning"):
                        st.write(result.get("reasoning",""))

                    meta_store = {
                        "route":      route,
                        "confidence": result["confidence"],
                        "reasoning":  result.get("reasoning",""),
                        "pipeline":   result["metadata"].get("pipeline",""),
                    }

                    if route == "LOCAL-TRAVERSAL" and G_viz:
                        sources = result.get("sources", [])
                        seed_entities  = result["metadata"].get("seed_entities", [])
                        subgraph_nodes = sources if isinstance(sources, list) else []

                        meta_store["traversal"] = {
                            "seed_entities":  seed_entities,
                            "subgraph_nodes": subgraph_nodes,
                        }

                        with st.expander("🕸 Traversed Subgraph", expanded=True):
                            st.caption(f"**Query entities (red):** {', '.join(seed_entities)}")
                            render_traversal_graph(
                                G_viz,
                                seed_nodes=set(seed_entities),
                                subgraph_nodes=set(subgraph_nodes),
                            )

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "thinking": thinking if 'thinking' in locals() else "",
                        "meta":    meta_store,
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    status_box.update(label="Error occurred", state="error")
                finally:
                    for lname in loggers:
                        logging.getLogger(lname).removeHandler(log_handler)

    if st.session_state.messages:
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE 2 — INGEST
# ══════════════════════════════════════════════════════════════
elif page == "📄 Document Ingestion":
    st.info("Upload documents to build your Vector Store and Knowledge Graph.")
    uploaded = st.file_uploader("Drop PDFs or Text files here", type=["pdf","txt"], accept_multiple_files=True)

    if uploaded and st.button("🚀 Process Documents"):
        from src.chunker import chunk_document
        from src.vector_store import VectorStore
        vs         = VectorStore()
        all_chunks = []
        progress   = st.progress(0)

        for i, f in enumerate(uploaded):
            st.write(f"Processing: {f.name}")
            if f.name.endswith(".pdf"):
                import fitz
                doc  = fitz.open(stream=f.read(), filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
            else:
                text = f.read().decode("utf-8")

            doc_id = f.name.replace(" ","_").replace(".pdf","").replace(".txt","")
            chunks = chunk_document(text, doc_id=doc_id)
            all_chunks.extend(chunks)
            progress.progress((i+1) / len(uploaded))

        st.write(f"Indexing {len(all_chunks)} chunks...")
        vs.build_index(all_chunks)
        st.success(f"Successfully Indexed {len(all_chunks)} chunks!")
        st.cache_resource.clear()

# ══════════════════════════════════════════════════════════════
# PAGE 3 — KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════
elif page == "🕸 Graph Explorer":
    G = load_graph()
    if G is None:
        st.warning("No knowledge graph found. Run the ingestion pipeline first.")
    else:
        st.markdown("""
        <div style="display: flex; justify-content: space-around; background: #ffffff; border: 1px solid #e2e8f0;; padding: 15px; border-radius: 12px; margin-bottom: 20px;">
            <div style="text-align: center;"><div style="color:#64748b;">Total Nodes</div><div style="font-size: 2rem; color: #10b981; font-weight:bold;">{}</div></div>
            <div style="text-align: center;"><div style="color:#64748b;">Total Edges</div><div style="font-size: 2rem; color: #059669; font-weight:bold;">{}</div></div>
            <div style="text-align: center;"><div style="color:#64748b;">Components</div><div style="font-size: 2rem; color: #059669; font-weight:bold;">{}</div></div>
        </div>
        """.format(G.number_of_nodes(), G.number_of_edges(), nx.number_weakly_connected_components(G)), unsafe_allow_html=True)

        try:
            from pyvis.network import Network
            import streamlit.components.v1 as components
            import tempfile, os

            TYPE_COLORS = {
                "model":     "#FF6B6B",
                "algorithm": "#4ECDC4",
                "technique": "#45B7D1",
                "concept":   "#96CEB4",
                "dataset":   "#FFEAA7",
                "metric":    "#DDA0DD",
                "task":      "#98D8C8",
            }

            net = Network(height="800px", width="100%",
                          bgcolor="#0b0f19", font_color="#ffffff",
                          directed=True)

            for node, data in G.nodes(data=True):
                etype = data.get('entity_type','concept')
                freq  = data.get('frequency', 1)
                color = TYPE_COLORS.get(etype, "#888888")
                title = f"{node}\nType: {etype}\nFreq: {freq}"
                net.add_node(node, label=node, color=color,
                             size=10 + freq*3, title=title,
                             font={"size": 12, "face": "Inter", "color": "#fff"})

            for src, tgt, data in G.edges(data=True):
                w   = data.get("computed_weight", 1)
                net.add_edge(src, tgt, label=data.get("relation_type",""),
                             width=min(w/3, 3),
                             color="rgba(100,120,150,0.5)", arrows="to")

            net.set_options('''{
                "physics": {
                    "forceAtlas2Based": {"gravitationalConstant": -60},
                    "solver": "forceAtlas2Based",
                    "stabilization": {"iterations": 200}
                },
                "interaction": {
                    "hover": true, 
                    "zoomView": true,
                    "dragView": true,
                    "zoomSpeed": 0.2,
                    "navigationButtons": true
                }
            }''')

            with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as f:
                net.save_graph(f.name)
                html_path = f.name
            with open(html_path) as f:
                html = f.read()
            os.unlink(html_path)
            
            st.markdown("<div style='background: rgba(10, 15, 25, 0.5); border-radius: 12px; border: 1px solid #ffffff; border: 1px solid #e2e8f0;;'>", unsafe_allow_html=True)
            components.html(html, height=820)
            st.markdown("</div>", unsafe_allow_html=True)

        except ImportError:
            st.info("pip install pyvis for interactive graph")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — COMMUNITIES
# ══════════════════════════════════════════════════════════════
elif page == "🗂 Communities":
    from src.config import SUMMARIES_DIR

    sel_res      = st.selectbox("Clustering Resolution Level", [0.5, 1.0, 2.0], index=1)
    summary_file = SUMMARIES_DIR / f"all_summaries_r{sel_res}.json"

    if not summary_file.exists():
        st.warning(f"No summaries found for resolution={sel_res}.")
    else:
        with open(summary_file) as f:
            summaries = json.load(f)

        sorted_comms = sorted(summaries.items(), key=lambda x: x[1].get("n_members",0), reverse=True)

        for comm_id, data in sorted_comms:
            members = data.get("members",[])
            with st.expander(f"🧩 Community {comm_id} ({len(members)} entities)"):
                st.markdown(f"<div style='background:#ffffff; border: 1px solid #e2e8f0;; padding:15px; border-radius:8px;'>{data.get('summary','')}</div>", unsafe_allow_html=True)
                st.caption(f"**Members:** {', '.join(members)}")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT PROJECT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.title("ℹ️ About GraphRAG Studio")
    
    st.markdown('''
    Welcome to **GraphRAG Studio**, an advanced implementation of Graph Retrieval-Augmented Generation. 
    This system goes beyond traditional vector-based RAG by constructing a semantic knowledge graph from your documents, allowing the LLM to understand complex relationships, global contexts, and multi-hop connections that simple text-chunk retrieval often misses.
    ''')
    
    st.header("Pipeline Architecture", divider="green")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
        ### 1. Document Ingestion
        Raw documents (PDFs, TXT, etc.) are ingested and broken down into smaller, manageable text chunks. These chunks are embedded using high-dimensional vectors and stored in a **ChromaDB** vector store for fast semantic similarity search.
        ''')
    with col2:
        st.markdown('''
        ### 2. Graph Extraction
        Simultaneously, the LLM processes these chunks to extract **Entities** (e.g., people, places, concepts) and **Relationships** between them. This structured data forms a rich Knowledge Graph stored locally using NetworkX.
        ''')
        
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('''
        ### 3. Community Detection
        We use the **Hierarchical Leiden Algorithm** to detect communities (clusters of highly connected entities) within the graph. This allows the system to understand macro-level themes across the entire dataset.
        ''')
    with col4:
        st.markdown('''
        ### 4. Community Summarization
        Once communities are detected, the LLM generates a comprehensive summary for each community. These summaries act as high-level reports that answer broad, global queries about the dataset.
        ''')

    st.header("Intelligent Query Routing", divider="green")
    st.info('''
    When you ask a question in the **RAG Chat**, our Intelligent Query Router analyzes the intent of your question and dynamically selects the best retrieval strategy:
    ''')
    
    st.markdown('''
    - 🔵 **LOCAL-VECTOR (Targeted Search)**: Best for specific, fact-based queries (e.g., *"What is the revenue for Q3?"*). It retrieves the most relevant raw text chunks using ChromaDB.
    - 🟣 **LOCAL-TRAVERSAL (Graph Walking)**: Best for multi-hop relationship queries (e.g., *"Who is the CEO connected to?"*). It finds a starting entity in the graph and traverses its neighbors to build a subgraph of context.
    - 🟢 **GLOBAL (Map-Reduce)**: Best for broad, thematic queries (e.g., *"What are the main challenges mentioned in these documents?"*). It aggregates all Community Summaries, asks the LLM to answer the question using each summary, and then reduces the intermediate answers into a final comprehensive response.
    - 🟠 **HYBRID (Combined)**: Fuses both Vector chunks and Graph summaries for maximum context.
    ''')
    
    st.header("Technology Stack", divider="green")
    st.markdown('''
    - **Frontend:** Streamlit with Custom CSS (Green-White Aesthetic)
    - **Graph Processing:** NetworkX, Graspologic (Leiden algorithm)
    - **Vector Store:** ChromaDB
    - **LLM Engine:** Groq (Llama-3) / Local Ollama (Qwen)
    - **Embeddings:** Local BGE/Nomic Embeddings via Ollama
    ''')
