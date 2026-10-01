"""SQL deep agent (LangChain `deepagents` + Ollama) — Streamlit chat app.

Run with:
    uv add streamlit deepagents langchain-ollama langchain-community mlflow
    uv run streamlit run 01_deep_agent.py

View traces with:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import uuid

import mlflow
import streamlit as st

st.set_page_config(page_title="SQL Deep Agent (LangChain + Ollama)", layout="wide")

mlflow.set_experiment("agents-workshop")
mlflow.langchain.autolog()

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

from deepagents import create_deep_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

def build_agent(model: str, base_url: str, db_uri: str, max_steps: int):
    """Configure a LangChain deep agent over a SQL database and return (agent, table_names)."""

    llm = ChatOllama(model=model, base_url=base_url, validate_model_on_init=True, temperature=0)

    db = SQLDatabase.from_uri(db_uri)
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=INSTRUCTIONS,
        checkpointer=InMemorySaver(),
    )
    st.session_state["recursion_limit"] = max_steps
    tool_names = [tool.name for tool in tools]
    return agent, tool_names, db.get_usable_table_names()


def render_trajectory(trajectory):
    if not trajectory:
        return
    with st.expander("Agent trajectory"):
        for msg in trajectory:
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                st.markdown(f"Tool Call: `{calls}`")
            else:
                st.markdown(f"{msg.type}: {msg.content or msg.additional_kwargs}")


# --- Sidebar: configuration ---------------------------------------------------
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
        st.session_state.setdefault("thread_id", str(uuid.uuid4()))
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")


# --- Main: chat ---------------------------------------------------------------
st.title("SQL Deep Agent")
st.session_state.setdefault("messages", [])

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_trajectory(msg.get("trajectory"))

if not st.session_state.get("agent_ready"):
    st.info("Set the database URI in the sidebar, then click **Connect**.")
elif question := st.chat_input("Ask a question about the database…"):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        trajectory = []
        run_config = {
            "recursion_limit": st.session_state.get("recursion_limit", 50),
            "configurable": {"thread_id": st.session_state["thread_id"]},
        }
        with st.spinner("Thinking…"):
            try:
                prior_state = st.session_state["agent"].get_state(run_config)
                prior_len = len(prior_state.values.get("messages", []))

                final_state = st.session_state["agent"].invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config=run_config,
                )
                trajectory = final_state.get("messages", [])[prior_len:]
            except Exception as e:
                st.exception(e)

        answer = trajectory[-1].content if trajectory else None
        if answer:
            st.markdown(answer)
            render_trajectory(trajectory)
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "trajectory": trajectory}
            )
