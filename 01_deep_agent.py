"""SQL deep agent (LangChain `deepagents` + Ollama) — Streamlit chat app.

Run with:
    uv add streamlit deepagents langchain-ollama langchain-community
    uv run streamlit run 01_deep_agent.py
"""

import uuid

import streamlit as st
from deepagents import create_deep_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

st.set_page_config(page_title="SQL Deep Agent (LangChain + Ollama)", layout="wide")

INSTRUCTIONS = """You are an agent designed to interact with a SQL database.

Given an input question, create a syntactically correct sqlite query, run it,
inspect the results, and return a concise answer. Unless the user asks for a
specific number of rows, limit results to at most 10. Never `SELECT *` — only
request the columns you need.

Do NOT issue any DML statements (INSERT, UPDATE, DELETE, DROP, etc.).

Always verify table names with `sql_db_list_tables` and inspect schemas with
`sql_db_schema` before writing a query. Use `sql_db_query_checker` on any
non-trivial query before running it with `sql_db_query`.

For multi-step questions, use your todo list to plan the steps before you
start querying.

Answer in plain language. Do not describe the database schema or the steps
you took.
"""


# ============================================================================
# Agent — building the deep agent and driving it. No Streamlit calls here.
# ============================================================================

def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def build_agent(model: str, base_url: str, db_uri: str):
    """Configure a LangChain deep agent over a SQL database and return (agent, tool_names, table_names)."""

    llm = ChatOllama(model=model, base_url=base_url, validate_model_on_init=True, temperature=0)

    db = SQLDatabase.from_uri(db_uri)
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=INSTRUCTIONS,
        checkpointer=InMemorySaver(),
    )
    tool_names = [tool.name for tool in tools]
    return agent, tool_names, db.get_usable_table_names()


def run_agent_turn(agent, question: str, run_config: dict, *, on_reasoning=None, on_text=None):
    """Drive one turn of the agent, invoking callbacks with streamed deltas.

    Returns the list of new messages produced by this turn (the trajectory).
    """
    prior_state = agent.get_state(run_config)
    prior_len = len(prior_state.values.get("messages", []))

    run = agent.stream_events(
        {"messages": [{"role": "user", "content": question}]},
        version="v3",
        config=run_config,
    )
    for message in run.messages:
        for reasoning_delta in message.reasoning:
            if on_reasoning:
                on_reasoning(reasoning_delta)
        for delta in message.text:
            if on_text:
                on_text(delta)

    final_state = run.output
    return (final_state or {}).get("messages", [])[prior_len:]


def extract_final_answer(trajectory) -> str | None:
    for msg in reversed(trajectory):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return _content_text(msg.content)
    return None


# ============================================================================
# UI — Streamlit rendering and session-state management.
# ============================================================================

def render_sidebar():
    """Render the configuration sidebar and return (model, base_url, db_uri, max_steps, connect_clicked)."""
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

        st.subheader("Database")
        db_uri = st.text_input(
            "SQLAlchemy URI",
            value="sqlite:///Chinook.db",
            help="e.g. sqlite:///Chinook.db, postgresql://user:pw@host/db",
        )

        max_steps = st.slider("Max recursion steps", 5, 100, 50)

        col_connect, col_clear = st.columns(2)
        connect_clicked = col_connect.button("Connect", type="primary", use_container_width=True)
        if col_clear.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["thread_id"] = str(uuid.uuid4())

        if st.session_state.get("agent_ready"):
            st.success(f"Connected — {st.session_state.get('connected_model')}")
            with st.expander(f"Tools ({len(st.session_state.get('tool_names', []))})"):
                st.write(st.session_state.get("tool_names", []))
            with st.expander(f"Tables ({len(st.session_state.get('tables', []))})"):
                st.write(st.session_state.get("tables", []))

    return model, base_url, db_uri, max_steps, connect_clicked


def connect_agent(model: str, base_url: str, db_uri: str, max_steps: int):
    """Build the agent from the sidebar config and store it in session state."""
    try:
        with st.spinner("Connecting…"):
            agent, tool_names, tables = build_agent(model, base_url, db_uri)
        st.session_state["agent"] = agent
        st.session_state["tool_names"] = tool_names
        st.session_state["tables"] = tables
        st.session_state["connected_model"] = model
        st.session_state["recursion_limit"] = max_steps
        st.session_state["agent_ready"] = True
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("thread_id", str(uuid.uuid4()))
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")


def render_chat_history():
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_agent_turn(question: str):
    """Stream one agent turn into the chat UI and return (answer, trajectory)."""
    text_buffer = ""
    reasoning_buffer = ""
    trajectory = []
    run_config = {
        "recursion_limit": st.session_state.get("recursion_limit", 50),
        "configurable": {"thread_id": st.session_state["thread_id"]},
    }

    with st.status("Thinking…", expanded=False) as status:
        reasoning_log = st.empty()
        log = st.empty()

        def on_reasoning(delta: str):
            nonlocal reasoning_buffer
            reasoning_buffer += delta
            reasoning_log.caption(f"🤔 {reasoning_buffer[-1000:]}")

        def on_text(delta: str):
            nonlocal text_buffer
            text_buffer += delta
            log.code(text_buffer[-2000:], language="text")

        try:
            trajectory = run_agent_turn(
                st.session_state["agent"],
                question,
                run_config,
                on_reasoning=on_reasoning,
                on_text=on_text,
            )
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")
            st.exception(e)

    return extract_final_answer(trajectory) or text_buffer, trajectory


def render_trajectory(trajectory):
    if not trajectory:
        return
    with st.expander("Agent trajectory"):
        for msg in trajectory:
            st.text(f"{msg.type}: {_content_text(msg.content) or msg.additional_kwargs}")


def handle_question(question: str):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer, trajectory = render_agent_turn(question)
        if answer:
            st.markdown(answer)
            render_trajectory(trajectory)
            st.session_state["messages"].append({"role": "assistant", "content": answer})


# --- App -----------------------------------------------------------------
model, base_url, db_uri, max_steps, connect_clicked = render_sidebar()
if connect_clicked:
    connect_agent(model, base_url, db_uri, max_steps)

st.title("SQL Deep Agent")
st.session_state.setdefault("messages", [])
render_chat_history()

if not st.session_state.get("agent_ready"):
    st.info("Set the database URI in the sidebar, then click **Connect**.")
elif question := st.chat_input("Ask a question about the database…"):
    handle_question(question)
