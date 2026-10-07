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
import hashlib  # fingerprints (pdf bytes, url, embedding model) into a db folder name
import os  # path joins, checking whether a Chroma folder already exists
import tempfile  # spools an uploaded PDF to a real path PyPDFLoader can open
import uuid  # per-connection thread_id for the checkpointer
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

NEVER answer from your knowledge.

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
  search, not the indexed documents. Add the link if available.

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
# Confirmed by actually crawling docs.incorta.com: of 36 URLs the loader
# followed, this list is what separates the 15 real pages from the other 21
# (a sitemap.xml, webfonts, a PNG icon, ...).
_SKIP_URL_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".css", ".js",
    ".xml", ".json", ".webmanifest", ".pdf",
)


# ----------------------------------------------------------------------------
# RAG
# ----------------------------------------------------------------------------

class RAGTools:
    """Owns the combined PDF + crawled-site Chroma index and exposes it to
    the agent as a single search tool. All the indexing logic — reusing an
    existing on-disk collection vs. building one from scratch — lives in
    __init__, so by the time get_tools() is called the index is ready."""

    def __init__(self, pdf_path: str, pdf_bytes: bytes, crawl_url: str,
                 embedding_model: str, base_url: str, 
                 chunk_size: int = 1000, chunk_overlap: int = 150):
        # Kept around only so rag_search's citation() can fall back to it if
        # a chunk's own metadata is somehow missing a 'source' — see below.
        self.pdf_path = pdf_path
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)

        # Index — keyed by (pdf, url, embedding model) so the expensive part
        # (loading the PDF, crawling the site, and embedding every chunk —
        # one Ollama call each) only ever happens once per distinct trio.
        # Reconnecting with the exact same PDF, URL, and model just reopens
        # the Chroma collection already on disk; changing any one of them
        # computes a new path and builds a fresh collection instead of
        # appending to (or confusing itself with) the old one.
        db_path = self._db_path(pdf_bytes, crawl_url, embedding_model)
        # Chroma(...) itself is cheap and safe either way: with
        # create_collection_if_not_exists=True (the default), it opens the
        # collection already on disk at db_path, or creates an empty one if
        # db_path doesn't exist yet — no PDF read, no crawl, no embedding
        # calls happen just from constructing it.
        self.vectorstore = Chroma(
            collection_name="docs",
            persist_directory=db_path,
            embedding_function=embeddings,
            create_collection_if_not_exists=True,  # Defaults to True
        )
        # Index documents if the current collection is empty.
        if self.num_chunks == 0:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            # One Document per PDF page (so page numbers survive into chunk
            # metadata) plus one per crawled page (URL as its own citation).
            # Splitting both the same way means rag_search's
            # similarity_search() below never has to know which source a
            # given chunk came from.
            pdf_chunks = splitter.split_documents(PyPDFLoader(pdf_path).load())
            web_chunks = splitter.split_documents(self._crawl_docs(crawl_url))
            self.vectorstore.add_documents(pdf_chunks + web_chunks)

    @staticmethod
    def _db_path(pdf_bytes: bytes, crawl_url: str, embedding_model: str) -> str:
        """A data/ path keyed by a hash of (PDF content, crawl URL, embedding
        model) — stable across reconnects with the same trio, but a fresh
        path (so a fresh build, never a stale mix) the moment any one of
        them changes. The embedding model matters as much as the content:
        two models' vectors aren't comparable, so reusing an old collection
        after switching models would silently return nonsense similarity
        scores."""
        key = pdf_bytes + crawl_url.encode() + embedding_model.encode()
        digest = hashlib.sha256(key).hexdigest()[:16]
        return os.path.join("data", f"chroma_{digest}")

    @staticmethod
    def _crawl_docs(url: str):
        """Crawl `url` and every page reachable within 2 links of it
        (RecursiveUrlLoader's own default max_depth), extracting plain text
        from each page's HTML and dropping non-page assets the crawler's
        link-following also turns up.

        RecursiveUrlLoader fetches `url` itself, regex-finds every href/src
        on the page, recurses into each one up to max_depth hops, and runs
        `extractor` over each page's raw HTML to turn it into a Document
        (one per URL, with `source` = that URL in its metadata).
        `prevent_outside` defaults to True, so it never wanders off
        docs.incorta.com onto some external link it happens to find.
        """
        # BeautifulSoup just strips tags/scripts/styles down to visible text;
        # get_text(separator=" ") keeps words from different elements from
        # running together (e.g. a nav link glued straight onto a heading).
        extractor = lambda html: bs4.BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
        loader = RecursiveUrlLoader(url, max_depth=2, extractor=extractor, timeout=10)
        return [
            doc for doc in loader.load()
            # Strip any query string before checking the extension, so
            # "icon.png?v=abc123" (a real URL the crawl turned up) is still
            # recognized as a PNG and filtered out.
            if not doc.metadata.get("source", "").split("?")[0].lower().endswith(_SKIP_URL_EXTENSIONS)
        ]

    def rag_search(self, query: str) -> str:
        """Search the indexed PDF and documentation pages for passages
        relevant to the question.

        Args:
            query: What to search for.

        Always try this before duckduckgo_search. Each result below is
        tagged with its source — a file name + page number, or a URL —
        quote that citation in your final answer.
        """
        # k=4: embed the query, return the 4 closest chunks by vector
        # distance — could be all 4 from the PDF, all 4 from the crawled
        # site, or any mix; the vectorstore doesn't distinguish by source.
        results = self.vectorstore.similarity_search(query, k=4)
        if not results:
            return "No relevant passages found in the indexed documents."

        def citation(doc):
            # Web chunks: PyPDFLoader never produced them, so 'source' is
            # just the page URL RecursiveUrlLoader set as metadata — quote
            # it as-is, there's no page number to add.
            source = doc.metadata.get("source", "")
            if source.startswith("http://") or source.startswith("https://"):
                return f"[Source: {source}]"
            # PDF chunks: 'source' is the file path PyPDFLoader was given
            # (basename only, so a temp upload path like /tmp/tmpXYZ.pdf
            # doesn't leak into the citation); 'page_label' is the page
            # number as printed on the page (falls back to the 0-indexed
            # 'page' if a PDF has no page_label metadata at all).
            page = doc.metadata.get("page_label", doc.metadata.get("page"))
            return f"[Source: {os.path.basename(source or self.pdf_path)}, page {page}]"

        # Each result keeps its own bracketed citation directly above its
        # text, so whichever chunks the model reads, the citation it needs
        # to quote is sitting right next to them — not a separate lookup.
        return "\n\n".join(f"{citation(doc)}\n{doc.page_content}" for doc in results)

    def get_tools(self) -> list:
        """The bound method bind_tools() (and the agent's dispatch) should
        see — `self` is already filled in, so it reads like a free function."""
        return [self.rag_search]

    @property
    def num_chunks(self) -> int:
        """Total chunks in the index — _collection.count() is chromadb's own
        native size check, whether the collection was just built or reopened."""
        return self.vectorstore._collection.count()


# ----------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------

class DuckDuckGoTools:
    """Wraps the web-search fallback — the one tool rag_search can't cover,
    for questions neither the PDF nor the crawled site answers."""

    def __init__(self):
        self._search = DuckDuckGoSearchRun()

    def get_tools(self) -> list:
        return [self._search]


def build_agent(
    model: str, base_url: str, pdf_path: str, pdf_bytes: bytes, crawl_url: str, embedding_model: str, max_steps: int
):
    """Configure a LangChain deep agent over a combined PDF + crawled-site
    Chroma index and return (agent, tool_names, num_chunks)."""

    # 1. Model — reasons and writes the final answer. temperature=0 keeps
    # answers and source citations consistent; validate_model_on_init fails
    # fast if it isn't pulled yet. (RAGTools builds its own embeddings model
    # separately — a different Ollama model doing a different job, turning
    # text into vectors rather than writing answers.)
    llm = ChatOllama(model=model, base_url=base_url, validate_model_on_init=True, temperature=0)

    # 2. Tools — RAGTools' __init__ does all the indexing work (reuse an
    # existing on-disk collection, or build one from scratch) before this
    # line returns; DuckDuckGoTools just wraps the one ready-made search tool.
    rag = RAGTools(pdf_path, pdf_bytes, crawl_url, embedding_model, base_url)
    duckduckgo = DuckDuckGoTools()
    tools = rag.get_tools() + duckduckgo.get_tools()

    # 3 & 4. Prompt + memory, assembled: create_deep_agent wraps the model,
    # tools, system prompt (above), and checkpointer into a LangGraph ReAct
    # loop — with planning and a virtual file system built in.
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=INSTRUCTIONS,
        checkpointer=InMemorySaver(),  # in-memory per-thread message history
    )
    st.session_state["recursion_limit"] = max_steps
    # rag_search is a bound method (name via .__name__, same as a free
    # function); DuckDuckGoSearchRun is a BaseTool (name via .name) — handle both.
    tool_names = [getattr(tool, "name", getattr(tool, "__name__", str(tool))) for tool in tools]
    return agent, tool_names, rag.num_chunks


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
