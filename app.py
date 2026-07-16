"""
app.py — GraphRAG Streamlit interface with traversal visualization.

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

# ── Point to v2 multi-doc graph ───────────────────────────────
V2_GRAPH = Path('outputs/graphs/v2_multi_doc/knowledge_graph.gml')
if V2_GRAPH.exists():
    # Copy v2 partitions to main folder so load_partitions finds them
    for f in glob.glob('outputs/graphs/v2_multi_doc/partition_*.json'):
        dst = 'outputs/graphs/' + os.path.basename(f)
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
    # Copy v2 summaries
    v2_summ = 'outputs/summaries/v2_multi_doc/all_summaries_r1.0.json'
    main_summ = 'outputs/summaries/all_summaries_r1.0.json'
    if os.path.exists(v2_summ) and not os.path.exists(main_summ):
        shutil.copy2(v2_summ, main_summ)
    # Override config to use v2 graph
    import sys
    sys.path.insert(0, '.')
    from src.config import cfg
    cfg['graph']['path'] = str(V2_GRAPH)

st.set_page_config(
    page_title="GraphRAG Chatbot",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load engines ──────────────────────────────────────────────
@st.cache_resource
def load_engine():
    from src.llm_client import LLMClient
    from src.vector_store import VectorStore
    from src.router import QueryRouter
    from src.query_engine import QueryEngine
    from src.config import GRAPH_PATH

    llm    = LLMClient(purpose="generation")
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
            st.sidebar.warning(f"Graph not loaded: {e}")

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
def render_traversal_graph(G, seed_nodes, subgraph_nodes):
    """
    Render a highlighted subgraph showing traversal.
    Seed nodes = red (query entities)
    Traversed nodes = orange
    Rest = grey (not shown)
    """
    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components
        import tempfile, os

        if not subgraph_nodes:
            return

        subG = G.subgraph(list(subgraph_nodes))
        net  = Network(height="400px", width="100%",
                       bgcolor="#1a1a2e", font_color="#ffffff",
                       directed=True)

        for node in subG.nodes():
            if node in seed_nodes:
                # Seed = large red node (query entity)
                color = "#FF4B4B"
                size  = 35
                title = f"🔴 QUERY ENTITY: {node}\n{G.nodes[node].get('description','')[:100]}"
            else:
                # Traversed = orange node
                color = "#FF8C00"
                size  = 20
                title = f"🟠 {node}\n{G.nodes[node].get('description','')[:100]}"

            net.add_node(node, label=node, color=color, size=size,
                         title=title, font={"size": 12})

        for src, tgt, data in subG.edges(data=True):
            rel = data.get("relation_type", "")
            net.add_edge(src, tgt, label=rel,
                         color="#888888", arrows="to",
                         title=data.get("description",""))

        net.set_options('''{
            "physics": {"stabilization": {"iterations": 150}},
            "edges": {"font": {"size": 9, "color": "#aaaaaa"}},
            "interaction": {"hover": true}
        }''')

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as f:
            net.save_graph(f.name)
            html_path = f.name

        with open(html_path) as f:
            html = f.read()
        os.unlink(html_path)

        st.caption("🕸 **Graph traversal** — red = query entities, orange = traversed nodes")
        components.html(html, height=420)

    except ImportError:
        # Fallback: matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if not subgraph_nodes:
            return

        subG = G.subgraph(list(subgraph_nodes))
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        pos    = nx.spring_layout(subG, seed=42, k=2)
        colors = ["#FF4B4B" if n in seed_nodes else "#FF8C00"
                  for n in subG.nodes()]

        nx.draw_networkx_nodes(subG, pos, node_color=colors,
                               node_size=800, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(subG, pos, font_size=7,
                                font_color='white', ax=ax)
        nx.draw_networkx_edges(subG, pos, edge_color='#888888',
                               arrows=True, ax=ax, alpha=0.6)

        red_patch    = mpatches.Patch(color='#FF4B4B', label='Query entity')
        orange_patch = mpatches.Patch(color='#FF8C00', label='Traversed node')
        ax.legend(handles=[red_patch, orange_patch], loc='upper left',
                  facecolor='#2a2a4e', labelcolor='white')
        ax.axis('off')
        ax.set_title("Graph Traversal", color='white', pad=10)
        st.pyplot(fig)


# ── Route badge color ─────────────────────────────────────────
ROUTE_COLORS = {
    "LOCAL-VECTOR":    "🔵",
    "LOCAL-TRAVERSAL": "🟣",
    "GLOBAL":          "🟢",
    "HYBRID":          "🟠",
}


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🔗 GraphRAG")
    st.caption("Graph-Based Retrieval-Augmented Generation")
    st.divider()

    st.subheader("Query settings")
    force_route = st.selectbox(
        "Force route",
        ["Auto (router decides)", "LOCAL", "LOCAL-TRAVERSAL", "GLOBAL", "HYBRID"],
    )
    force = None if force_route.startswith("Auto") else force_route

    st.divider()
    st.subheader("Index status")
    try:
        from src.vector_store import VectorStore
        from src.config import GRAPH_PATH
        vs    = VectorStore()
        stats = vs.stats()
        st.metric("Chunks indexed", stats["total_chunks"])
        st.metric("Graph exists", "✅ Yes" if GRAPH_PATH.exists() else "❌ No")
        if GRAPH_PATH.exists():
            G_side = nx.read_gml(str(GRAPH_PATH))
            st.metric("Graph nodes", G_side.number_of_nodes())
            st.metric("Graph edges", G_side.number_of_edges())
    except Exception:
        st.info("Not yet indexed")

    st.divider()
    st.caption("Route legend:\n🔵 Vector RAG\n🟣 Graph Traversal\n🟢 Global Map-Reduce\n🟠 Hybrid")


# ── Tabs ──────────────────────────────────────────────────────
tab_chat, tab_ingest, tab_graph, tab_communities = st.tabs([
    "💬 Chat", "📄 Ingest", "🕸 Knowledge Graph", "🗂 Communities"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Ask questions about your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    G_viz = load_graph()

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg["role"] == "assistant" and "meta" in msg:
                meta  = msg["meta"]
                route = meta.get("route", "?")
                emoji = ROUTE_COLORS.get(route, "⚪")

                col1, col2, col3 = st.columns(3)
                col1.metric("Route", f"{emoji} {route}")
                col2.metric("Confidence", f"{meta.get('confidence', 0):.0%}")
                col3.metric("Pipeline", meta.get("pipeline", "?"))

                with st.expander("Routing reasoning"):
                    st.caption(meta.get("reasoning", ""))

                # Show traversal graph if route was traversal
                if route == "LOCAL-TRAVERSAL" and G_viz and "traversal" in meta:
                    with st.expander("🕸 View traversed subgraph", expanded=True):
                        render_traversal_graph(
                            G_viz,
                            seed_nodes=set(meta["traversal"].get("seed_entities", [])),
                            subgraph_nodes=set(meta["traversal"].get("subgraph_nodes", [])),
                        )

    # Chat input
    if query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    engine = load_engine()
                    result = engine.query(query, force_route=force)
                    answer = result["answer"]
                    route  = result["route"]
                    emoji  = ROUTE_COLORS.get(route, "⚪")

                    st.write(answer)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Route", f"{emoji} {route}")
                    col2.metric("Confidence", f"{result['confidence']:.0%}")
                    col3.metric("Pipeline", result["metadata"].get("pipeline","?"))

                    with st.expander("Routing reasoning"):
                        st.caption(result.get("reasoning",""))

                    # Traversal visualization — show automatically
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

                        with st.expander("🕸 View traversed subgraph", expanded=True):
                            st.caption(f"**Query entities (red):** {', '.join(seed_entities)}")
                            st.caption(f"**Nodes reached in 2 hops (orange):** {len(subgraph_nodes)}")
                            render_traversal_graph(
                                G_viz,
                                seed_nodes=set(seed_entities),
                                subgraph_nodes=set(subgraph_nodes),
                            )

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "meta":    meta_store,
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    if st.session_state.messages:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 2 — INGEST
# ══════════════════════════════════════════════════════════════
with tab_ingest:
    st.subheader("Upload and index documents")
    st.info("Upload PDFs or TXT files. They will be chunked, embedded, and added to the vector index.")

    uploaded = st.file_uploader("Choose files", type=["pdf","txt"], accept_multiple_files=True)

    if uploaded and st.button("Index documents", type="primary"):
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
        st.success(f"Done! Indexed {len(all_chunks)} chunks from {len(uploaded)} files.")
        st.cache_resource.clear()


# ══════════════════════════════════════════════════════════════
# TAB 3 — KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════
with tab_graph:
    st.subheader("Knowledge graph explorer")

    G = load_graph()
    if G is None:
        st.warning("No knowledge graph found at outputs/graphs/knowledge_graph.gml")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", G.number_of_nodes())
        col2.metric("Edges", G.number_of_edges())
        col3.metric("Components", nx.number_weakly_connected_components(G))

        # Entity type breakdown
        from collections import Counter
        types = Counter(d.get('entity_type','?') for _, d in G.nodes(data=True))
        st.caption("Entity types: " + " · ".join(f"{t}: {n}" for t,n in types.most_common()))

        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)

        # Top entities table
        with st.expander("Top entities by connections", expanded=False):
            st.dataframe(
                [{"Entity": n, "Connections": d,
                  "Type": G.nodes[n].get('entity_type','?'),
                  "Frequency": G.nodes[n].get('frequency', 0)}
                 for n, d in degrees[:20]],
                use_container_width=True
            )

        # Full interactive graph
        st.subheader("Interactive knowledge graph")
        try:
            from pyvis.network import Network
            import streamlit.components.v1 as components
            import tempfile, os

            # Color by entity type
            TYPE_COLORS = {
                "model":     "#FF6B6B",
                "algorithm": "#4ECDC4",
                "technique": "#45B7D1",
                "concept":   "#96CEB4",
                "dataset":   "#FFEAA7",
                "metric":    "#DDA0DD",
                "task":      "#98D8C8",
            }

            net = Network(height="550px", width="100%",
                          bgcolor="#0e1117", font_color="#ffffff",
                          directed=True)

            for node, data in G.nodes(data=True):
                etype = data.get('entity_type','concept')
                freq  = data.get('frequency', 1)
                color = TYPE_COLORS.get(etype, "#888888")
                desc  = data.get('description','')[:120]
                claims = data.get('claims','')[:100]
                title = f"<b>{node}</b><br>Type: {etype}<br>Frequency: {freq}<br>{desc}"
                if claims: title += f"<br><i>{claims}</i>"
                net.add_node(node, label=node, color=color,
                             size=10 + freq*3, title=title,
                             font={"size": 11})

            for src, tgt, data in G.edges(data=True):
                rel = data.get("relation_type","")
                w   = data.get("computed_weight", 1)
                net.add_edge(src, tgt, label=rel,
                             width=min(w/5, 4),
                             color="#444444", arrows="to",
                             title=data.get("description",""))

            net.set_options('''{
                "physics": {
                    "forceAtlas2Based": {"gravitationalConstant": -50},
                    "solver": "forceAtlas2Based",
                    "stabilization": {"iterations": 200}
                },
                "interaction": {"hover": true, "tooltipDelay": 100}
            }''')

            # Legend
            legend_html = "<div style='font-size:12px;color:#ccc;margin-bottom:8px'>"
            for etype, color in TYPE_COLORS.items():
                legend_html += f"<span style='background:{color};padding:2px 8px;border-radius:10px;margin-right:5px;color:#000'>{etype}</span>"
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

            with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as f:
                net.save_graph(f.name)
                html_path = f.name
            with open(html_path) as f:
                html = f.read()
            os.unlink(html_path)
            components.html(html, height=570)

        except ImportError:
            st.info("pip install pyvis for interactive graph")


# ══════════════════════════════════════════════════════════════
# TAB 4 — COMMUNITIES
# ══════════════════════════════════════════════════════════════
with tab_communities:
    st.subheader("Community explorer")
    from src.config import SUMMARIES_DIR

    sel_res      = st.selectbox("Resolution level", [0.5, 1.0, 2.0], index=1)
    summary_file = SUMMARIES_DIR / f"all_summaries_r{sel_res}.json"

    if not summary_file.exists():
        st.warning(f"No summaries for resolution={sel_res}. Run community summarization first.")
    else:
        with open(summary_file) as f:
            summaries = json.load(f)

        st.metric("Communities", len(summaries))
        sorted_comms = sorted(summaries.items(),
                              key=lambda x: x[1].get("n_members",0), reverse=True)

        for comm_id, data in sorted_comms:
            members = data.get("members",[])
            summary = data.get("summary","No summary available")

            with st.expander(f"Community {comm_id} — {len(members)} entities"):
                st.write(summary)
                st.divider()
                # Show mini graph of this community
                G_comm = load_graph()
                if G_comm and len(members) > 1:
                    valid = [m for m in members if m in G_comm.nodes]
                    if len(valid) > 1:
                        sub = G_comm.subgraph(valid)
                        try:
                            from pyvis.network import Network
                            import streamlit.components.v1 as components
                            import tempfile, os
                            cnet = Network(height="250px", width="100%",
                                          bgcolor="#0e1117", font_color="#fff")
                            cnet.from_nx(sub)
                            with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as f:
                                cnet.save_graph(f.name)
                                cpath = f.name
                            with open(cpath) as f:
                                chtml = f.read()
                            os.unlink(cpath)
                            components.html(chtml, height=260)
                        except: pass
                st.caption("Members: " + ", ".join(members[:20]) +
                           (f" ... +{len(members)-20} more" if len(members)>20 else ""))
