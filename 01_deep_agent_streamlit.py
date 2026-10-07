"""SQL deep agent (LangChain `deepagents` + Ollama) — Streamlit chat app.

Run with:
    uv add streamlit deepagents langchain-ollama langchain-community mlflow ddgs
    uv run streamlit run 01_deep_agent.py

View traces with:
    uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db
"""

# --- Setup ---------------------------------------------------------------
import uuid

import mlflow
import streamlit as st

st.set_page_config(page_title="01 - SQL Deep Agent (LangChain + Ollama)", layout="wide")

# One call enables tracing for every LangChain/LangGraph call made below —
# each agent.invoke() becomes a trace with nested spans for the model and
# tool calls. View them with `mlflow ui --backend-store-uri sqlite:///data/mlflow.db`.
mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
mlflow.set_experiment("01_deep_agent")
mlflow.langchain.autolog()

# --- System prompt: steers the agent's ReAct loop (passed to create_deep_agent
# below as system_prompt). These four rules are what keep it from guessing at
# schemas, skipping validation, or running destructive SQL.
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

Always try the SQL tools first. If the data isn't in the database — either
the question is about something outside it (e.g. background on an artist)
or a query comes back empty — use `duckduckgo_search` to look it up on the
web instead of saying you don't know.

Answer in plain language. Do not describe the database schema or the steps
you took.
"""

# --- Core LangChain code: the four pieces that make up the agent ----------
from deepagents import create_deep_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

def build_agent(model: str, base_url: str, db_uri: str, max_steps: int):
    """Configure a LangChain deep agent over a SQL database and return (agent, table_names)."""

    # 1. Model — a local Ollama model. temperature=0 keeps SQL generation
    # deterministic; validate_model_on_init fails fast if it isn't pulled yet.
    llm = ChatOllama(model=model, base_url=base_url, validate_model_on_init=True, temperature=0)

    # 2. Tools — SQLDatabaseToolkit wraps the DB connection into four
    # ready-made LangChain tools: sql_db_list_tables, sql_db_schema,
    # sql_db_query_checker, sql_db_query. DuckDuckGoSearchRun adds a fifth,
    # for questions the database itself can't answer.
    db = SQLDatabase.from_uri(db_uri)
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools() + [DuckDuckGoSearchRun()]

    # 3 & 4. Prompt + memory, assembled: create_deep_agent wraps the model,
    # tools, system prompt (step 3 above), and checkpointer into a LangGraph
    # ReAct loop — with planning and a virtual file system built in.
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=INSTRUCTIONS,
        checkpointer=InMemorySaver(),  # in-memory per-thread message history
    )
    st.session_state["recursion_limit"] = max_steps
    tool_names = [tool.name for tool in tools]
    return agent, tool_names, db.get_usable_table_names()


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

    st.subheader("Database")
    db_uri = st.text_input(
        "SQLAlchemy URI",
        value="sqlite:///data/Chinook.db",
        help="e.g. sqlite:///data/Chinook.db, postgresql://user:pw@host/db",
    )

    max_steps = st.slider("Max recursion steps", 5, 100, 50)

    col_connect, col_clear = st.columns(2)
    connect_clicked = col_connect.button("Connect", type="primary", use_container_width=True)
    if col_clear.button("Clear chat", use_container_width=True):
        # A fresh thread_id makes the checkpointer start a brand-new
        # conversation on the next turn, instead of recalling the old one.
        st.session_state["messages"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())

    # Once connected, show what the agent has access to.
    if st.session_state.get("agent_ready"):
        st.success(f"Connected — {st.session_state.get('connected_model')}")
        with st.expander(f"Tools ({len(st.session_state.get('tool_names', []))})"):
            st.write(st.session_state.get("tool_names", []))
        with st.expander(f"Tables ({len(st.session_state.get('tables', []))})"):
            st.write(st.session_state.get("tables", []))


# Build (or rebuild) the agent and stash it in session_state — this is the
# only place build_agent() is called, so changing a sidebar setting has no
# effect until you click Connect again.
if connect_clicked:
    try:
        with st.spinner("Connecting…"):
            agent, tool_names, tables = build_agent(model, base_url, db_uri, max_steps)
        st.session_state["agent"] = agent
        st.session_state["tool_names"] = tool_names
        st.session_state["tables"] = tables
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
st.title("SQL Deep Agent")
st.session_state.setdefault("messages", [])

render_history()

# st.chat_input() is only called (and so only rendered) inside this elif, so
# the input box itself doesn't appear until an agent is connected.
if not st.session_state.get("agent_ready"):
    st.info("Set the database URI in the sidebar, then click **Connect**.")
elif question := st.chat_input("Ask a question about the database…"):
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
