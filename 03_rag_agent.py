"""RAG deep agent (LangChain `deepagents` + Ollama + Chroma) — Streamlit chat app.

Indexes both a local PDF and a crawl of a documentation site into one Chroma
vector store, persisted under data/ and keyed by a hash of (PDF bytes, crawl
URL) — so the same pair reopens the existing store instantly, and either one
changing builds a fresh store under a new path instead of mixing with stale data.

Run with:
    uv add streamlit deepagents langchain-ollama langchain-community mlflow ddgs \\
        langchain-chroma chromadb pypdf beautifulsoup4
    ollama pull qwen3-embedding:0.6b
    uv run streamlit run 03_rag_agent.py

View traces with:
    uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db
"""

# --- Setup ---------------------------------------------------------------
import hashlib
import os
import tempfile
import uuid
import warnings

import mlflow
import streamlit as st
from bs4 import XMLParsedAsHTMLWarning

# RecursiveUrlLoader's default extractor parses every crawled page (including
# the site's XML sitemaps it stumbles into) with an HTML parser — harmless
# here since sitemaps get filtered out below anyway, just noisy.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

st.set_page_config(page_title="03 - RAG Deep Agent (LangChain + Ollama + Chroma)", layout="wide")

# One call enables tracing for every LangChain/LangGraph call made below —
# each agent.invoke() becomes a trace with nested spans for the model and
# tool calls. View them with `mlflow ui --backend-store-uri sqlite:///data/mlflow.db`.
mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
mlflow.set_experiment("03_rag_agent")
mlflow.langchain.autolog()

# --- System prompt: steers the agent's ReAct loop (passed to create_deep_agent
# below as system_prompt). RAG first, web search as fallback, always cite a source.
INSTRUCTIONS = """You are a RAG (Retrieval-Augmented Generation) agent that answers
questions using an uploaded PDF and a crawled set of documentation pages,
both indexed into the same `rag_search` tool.

Always try `rag_search` first. Only use `duckduckgo_search` if neither the
PDF nor the documentation pages contain the answer — rag_search comes back
empty, or its passages are clearly unrelated to the question.

Whichever tool you used, always report the source of your answer:
- From `rag_search`, a PDF passage: end your answer with a line in exactly
  this form: `Source: <document name>, page <N>` (comma-separate multiple
  pages). Every result already carries this file name and page number —
  copy them verbatim, never guess or renumber.
- From `rag_search`, a documentation page: end your answer with
  `Source: <url>` — copy the URL verbatim from the result.
- From `duckduckgo_search`: say plainly that the answer came from a web
  search, not the indexed documents.

If none of the tools turn up an answer, say so instead of guessing.

Answer in plain language. Do not describe your search process beyond the
required source citation.
"""

# --- Core LangChain code: model, RAG index, tools, prompt + memory --------
import bs4
from deepagents import create_deep_agent
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, RecursiveUrlLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

# Non-page assets RecursiveUrlLoader's link-following turns up (icons, fonts,
# the site's own sitemap.xml, ...) — garbage once run through an HTML text
# extractor, so they're dropped by extension rather than indexed as "pages".
_SKIP_URL_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".css", ".js",
    ".xml", ".json", ".webmanifest", ".pdf",
)


def _db_path(pdf_bytes: bytes, crawl_url: str, embedding_model: str) -> str:
    """A data/ path keyed by a hash of (PDF content, crawl URL, embedding
    model) — stable across reconnects with the same trio, but a fresh path
    (so a fresh build, never a stale mix) the moment any one of them changes.
    The embedding model matters as much as the content: two models' vectors
    aren't comparable, so reusing an old collection after switching models
    would silently return nonsense similarity scores."""
    key = pdf_bytes + crawl_url.encode() + embedding_model.encode()
    digest = hashlib.sha256(key).hexdigest()[:16]
    return os.path.join("data", f"chroma_{digest}")


def _crawl_docs(url: str):
    """Crawl `url` and every page reachable within 2 links of it (RecursiveUrlLoader's
    own default max_depth), extracting plain text from each page's HTML and
    dropping non-page assets the crawler's link-following also turns up."""
    extractor = lambda html: bs4.BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    loader = RecursiveUrlLoader(url, max_depth=2, extractor=extractor, timeout=10)
    return [
        doc for doc in loader.load()
        if not doc.metadata.get("source", "").split("?")[0].lower().endswith(_SKIP_URL_EXTENSIONS)
    ]


def build_agent(
    model: str, base_url: str, pdf_path: str, pdf_bytes: bytes, crawl_url: str, embedding_model: str, max_steps: int
):
    """Configure a LangChain deep agent over a combined PDF + crawled-site
    Chroma index and return (agent, tool_names, num_chunks)."""

    # 1. Model — a local Ollama model. temperature=0 keeps answers and source
    # citations consistent; validate_model_on_init fails fast if it isn't pulled yet.
    llm = ChatOllama(model=model, base_url=base_url, validate_model_on_init=True, temperature=0)
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)

    # 2. Index — keyed by (pdf, url, embedding model) so the expensive part
    # (loading the PDF, crawling the site, and embedding every chunk — one
    # Ollama call each) only ever happens once per distinct trio. Reconnecting
    # with the exact same PDF, URL, and model just reopens the Chroma
    # collection already on disk; changing any one of them computes a new
    # path and builds a fresh collection instead of appending to (or
    # confusing itself with) the old one.
    db_path = _db_path(pdf_bytes, crawl_url, embedding_model)
    if os.path.isdir(db_path) and os.listdir(db_path):
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        # One Document per PDF page (so page numbers survive into chunk
        # metadata) plus one per crawled page (URL as its own citation).
        pdf_chunks = splitter.split_documents(PyPDFLoader(pdf_path).load())
        web_chunks = splitter.split_documents(_crawl_docs(crawl_url))
        vectorstore = Chroma.from_documents(pdf_chunks + web_chunks, embeddings, persist_directory=db_path)

    # 3. Tools — rag_search wraps the vectorstore as a retriever tool: every
    # result is tagged with a citation the model can quote directly — a file
    # name + page number for PDF passages, a URL for crawled pages.
    # duckduckgo_search is the fallback for questions neither one covers.
    def rag_search(query: str) -> str:
        """Search the indexed PDF and documentation pages for passages
        relevant to the question.

        Args:
            query: What to search for.

        Always try this before duckduckgo_search. Each result below is
        tagged with its source — a file name + page number, or a URL —
        quote that citation in your final answer.
        """
        results = vectorstore.similarity_search(query, k=4)
        if not results:
            return "No relevant passages found in the indexed documents."

        def citation(doc):
            source = doc.metadata.get("source", "")
            if source.startswith("http://") or source.startswith("https://"):
                return f"[Source: {source}]"
            page = doc.metadata.get("page_label", doc.metadata.get("page"))
            return f"[Source: {os.path.basename(source or pdf_path)}, page {page}]"

        return "\n\n".join(f"{citation(doc)}\n{doc.page_content}" for doc in results)

    tools = [rag_search, DuckDuckGoSearchRun()]

    # 4 & 5. Prompt + memory, assembled: create_deep_agent wraps the model,
    # tools, system prompt (above), and checkpointer into a LangGraph ReAct
    # loop — with planning and a virtual file system built in.
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=INSTRUCTIONS,
        checkpointer=InMemorySaver(),  # in-memory per-thread message history
    )
    st.session_state["recursion_limit"] = max_steps
    # rag_search is a plain function (name via __name__); DuckDuckGoSearchRun
    # is a BaseTool (name via .name) — handle both.
    tool_names = [getattr(tool, "name", getattr(tool, "__name__", str(tool))) for tool in tools]
    num_chunks = len(vectorstore.get(include=[])["ids"])
    return agent, tool_names, num_chunks


def render_trajectory(trajectory):
    """Render an expander listing the agent's steps for one turn: tool calls
    (name + args) and any plain message content, in the order they happened."""
    if not trajectory:
        return
    with st.expander("Agent trajectory"):
        for msg in trajectory:
            # msg.type is "human", "ai", or "tool"; an "ai" message additionally
            # carries tool_calls when the model chose to call a tool instead of
            # answering directly — color-code each case so the ReAct steps
            # (thought/action/observation) are easy to scan at a glance.
            if msg.type == "human":
                st.markdown(f":blue-badge[:material/person: Human] {msg.content}")
            elif msg.type == "tool":
                st.markdown(f":green-badge[:material/output: Tool result] `{msg.content}`")
            elif getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                st.markdown(f":orange-badge[:material/build: Tool call] `{calls}`")
            else:
                st.markdown(f":violet-badge[:material/smart_toy: AI] {msg.content or msg.additional_kwargs}")


def render_history():
    """Replay chat history on every rerun — Streamlit doesn't persist UI elements
    across reruns, so this (plus storing `trajectory` per message) is what makes
    past turns' answers and trajectories still visible after a new question."""
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_trajectory(msg.get("trajectory"))


# --- Sidebar: configuration + connection state -----------------------------
# Streamlit reruns this whole script top-to-bottom on every interaction, so
# widget values below are just read fresh each time; actually (re)building the
# agent only happens when the Connect button is clicked (see `if connect_clicked`).
with st.sidebar:
    st.header("Configuration")

    model = st.selectbox(
        "Model",
        [
            "qwen3.5:4b-mlx",
            "gemma4:e4b",
            "nemotron-3-nano:4b",
        ],
        index=0,
    )

    base_url = st.text_input(
        "Ollama base URL",
        value="http://localhost:11434",
    )

    embedding_model = st.selectbox(
        "Embedding model",
        [
            "qwen3-embedding:0.6b",
            "nomic-embed-text",
            "mxbai-embed-large",
        ],
        index=0,
        help="Changing this builds a fresh vector store — different models' embeddings aren't comparable.",
    )

    st.subheader("Document")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Leave empty to use the default document below.",
    )
    default_pdf_path = st.text_input(
        "...or a local PDF path",
        value="data/llms-text-generation.pdf",
        help="Used when no file is uploaded above.",
    )

    crawl_url = st.text_input(
        "Documentation URL",
        value="https://docs.incorta.com/latest",
        help="Crawled up to 2 links deep and indexed alongside the PDF.",
    )

    max_steps = st.slider("Max recursion steps", 5, 100, 50)

    col_connect, col_clear = st.columns(2)
    connect_clicked = col_connect.button("Connect", type="primary", width="stretch")
    if col_clear.button("Clear chat", width="stretch"):
        # A fresh thread_id makes the checkpointer start a brand-new
        # conversation on the next turn, instead of recalling the old one.
        st.session_state["messages"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())

    # Once connected, show what the agent has access to.
    if st.session_state.get("agent_ready"):
        st.success(f"Connected — {st.session_state.get('connected_model')}")
        with st.expander(f"Tools ({len(st.session_state.get('tool_names', []))})"):
            st.write(st.session_state.get("tool_names", []))
        num_chunks = st.session_state.get("num_chunks", 0)
        with st.expander(f"Indexed chunks ({num_chunks})"):
            st.write("PDF:", st.session_state.get("connected_pdf_name", ""))
            st.write("Crawled:", st.session_state.get("connected_crawl_url", ""))


# Build (or rebuild) the agent and stash it in session_state — this is the
# only place build_agent() is called, so changing a sidebar setting has no
# effect until you click Connect again.
if connect_clicked:
    try:
        with st.spinner("Indexing document + crawling site (first time only) and connecting…"):
            if uploaded_pdf is not None:
                # PyPDFLoader needs a real file path, so the uploaded bytes
                # are spooled to a temp file first; it stays on disk for the
                # lifetime of this connection (the vectorstore keeps its own
                # copy of the text, so the temp file itself is only read once).
                pdf_bytes = uploaded_pdf.getvalue()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes)
                    pdf_path = tmp.name
                pdf_name = uploaded_pdf.name
            else:
                pdf_path = default_pdf_path
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                pdf_name = os.path.basename(default_pdf_path)

            agent, tool_names, num_chunks = build_agent(
                model, base_url, pdf_path, pdf_bytes, crawl_url, embedding_model, max_steps
            )
        st.session_state["agent"] = agent
        st.session_state["tool_names"] = tool_names
        st.session_state["num_chunks"] = num_chunks
        st.session_state["connected_pdf_name"] = pdf_name
        st.session_state["connected_crawl_url"] = crawl_url
        st.session_state["connected_model"] = model
        st.session_state["agent_ready"] = True
        st.session_state.setdefault("messages", [])
        # thread_id is the checkpointer's conversation key — created once per
        # connection and reused across turns (see run_config below) so the
        # agent remembers earlier questions in the same session.
        st.session_state.setdefault("thread_id", str(uuid.uuid4()))
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")


# --- Main: chat -------------------------------------------------------------
st.title("RAG Deep Agent")
st.session_state.setdefault("messages", [])

render_history()

# st.chat_input() is only called (and so only rendered) inside this elif, so
# the input box itself doesn't appear until an agent is connected.
if not st.session_state.get("agent_ready"):
    st.info("Upload a PDF (or use the default) in the sidebar, then click **Connect**.")
elif question := st.chat_input("Ask a question about the document…"):
    # Draw the user's own message immediately, before calling the (slow)
    # agent below — render_history() already ran, so this turn's question
    # isn't in it yet.
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        trajectory = []
        # configurable.thread_id is how the checkpointer finds (and keeps
        # extending) this conversation's message history across turns.
        run_config = {
            "recursion_limit": st.session_state.get("recursion_limit", 50),
            "configurable": {"thread_id": st.session_state["thread_id"]},
        }
        with st.spinner("Thinking…"):
            try:
                # The checkpointer returns the FULL message history for this
                # thread_id (every prior turn), so grab its length first —
                # that's how we slice out just this turn's new messages below.
                prior_state = st.session_state["agent"].get_state(run_config)
                prior_len = len(prior_state.values.get("messages", []))

                # invoke() runs the agent's ReAct loop to completion: model ->
                # tool calls -> tool results -> model -> ... until the model
                # replies without calling a tool. final_state["messages"] is
                # the whole conversation so far, old turns included.
                final_state = st.session_state["agent"].invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config=run_config,
                )
                trajectory = final_state.get("messages", [])[prior_len:]
            except Exception as e:
                st.exception(e)

        # The agent only stops once the model replies with no tool calls, so
        # the last message in this turn's trajectory is always the answer.
        answer = trajectory[-1].content if trajectory else None
        # Only render/save a reply if the agent actually produced one — trajectory
        # stays empty when invoke() raised above, so there's nothing to show or
        # persist for this turn (the exception is already displayed via st.exception).
        if answer:
            st.markdown(answer)
            render_trajectory(trajectory)
            # Mirror what render_history() draws for a saved message, so this
            # turn looks identical to itself on the next rerun.
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "trajectory": trajectory}
            )
