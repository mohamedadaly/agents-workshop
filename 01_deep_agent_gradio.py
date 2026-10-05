"""SQL deep agent (LangChain `deepagents` + Ollama) — Gradio chat app.

Run with:
    uv add gradio deepagents langchain-ollama langchain-community mlflow ddgs
    uv run python 01_deep_agent_gradio.py

View traces with:
    uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db
"""

import uuid

import gradio as gr
import mlflow

# One call enables tracing for every LangChain/LangGraph call made below —
# each agent.invoke() becomes a trace with nested spans for the model and
# tool calls. View them with `mlflow ui --backend-store-uri sqlite:///data/mlflow.db`.
mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
mlflow.set_experiment("01_deep_agent")
mlflow.langchain.autolog()

# System prompt: steers the agent's ReAct loop (passed to create_deep_agent
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


def build_agent(model: str, base_url: str, db_uri: str):
    """Configure a LangChain deep agent over a SQL database and return (agent, tool_names, table_names)."""

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
    tool_names = [tool.name for tool in tools]
    return agent, tool_names, db.get_usable_table_names()


def describe_step(msg, parent_id):
    """Turn one trajectory message into a Gradio "thought" ChatMessage dict,
    nested under the single "Agent trajectory" thought via parent_id — so
    every step collapses into one accordion instead of one each. msg.type is
    "human", "ai", or "tool"; an "ai" message additionally carries tool_calls
    when the model chose to call a tool instead of answering directly.
    No "status" key: per Gradio, that's what leaves each step expanded by
    default once its parent is opened (a "done" status would collapse it)."""
    if msg.type == "tool":
        return {
            "role": "assistant",
            "content": f"`{msg.content}`",
            "metadata": {"title": "🟢 Tool result", "parent_id": parent_id},
        }
    if getattr(msg, "tool_calls", None):
        calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
        return {
            "role": "assistant",
            "content": f"`{calls}`",
            "metadata": {"title": "🟠 Tool call", "parent_id": parent_id},
        }
    # A plain AI message that isn't the final answer (rare, but possible).
    return {
        "role": "assistant",
        "content": str(msg.content or msg.additional_kwargs),
        "metadata": {"title": "🟣 AI", "parent_id": parent_id},
    }


def connect(model, base_url, db_uri):
    """Build (or rebuild) the agent and store it in Gradio's per-session state —
    this is the only place build_agent() is called, so changing a sidebar
    setting has no effect until Connect is clicked again."""
    try:
        agent, tool_names, tables = build_agent(model, base_url, db_uri)
    except Exception as e:
        return None, f"❌ Connection failed: {e}", gr.update(), gr.update(), gr.update(), gr.update()

    connection = {
        "agent": agent,
        "tool_names": tool_names,
        "tables": tables,
        "model": model,
        # thread_id is the checkpointer's conversation key — fresh on every
        # (re)connect so a new connection starts a clean conversation.
        "thread_id": str(uuid.uuid4()),
    }
    status = f"✅ **Connected — {model}**"
    return (
        connection,
        status,
        gr.update(label=f"Tools ({len(tool_names)})"),
        ", ".join(tool_names),
        gr.update(label=f"Tables ({len(tables)})"),
        ", ".join(tables),
    )


def clear_chat(connection):
    """A fresh thread_id makes the checkpointer start a brand-new conversation
    on the next turn, instead of recalling the old one — without rebuilding
    the (slow) agent connection itself."""
    if connection is not None:
        connection = {**connection, "thread_id": str(uuid.uuid4())}
    return connection, []


def respond(message, history, connection, max_steps):
    """Handle one chat turn and return the trajectory (as collapsible thoughts)
    plus the final answer. gr.ChatInterface appends the user's message and
    whatever we return here to the chat history itself — and clears the
    input box — so no manual history bookkeeping is needed; returning a list
    makes it append each element as its own message."""
    if connection is None:
        return "Set the database URI in the sidebar, then click **Connect**."

    agent = connection["agent"]
    # configurable.thread_id is how the checkpointer finds (and keeps
    # extending) this conversation's message history across turns.
    run_config = {
        "recursion_limit": max_steps,
        "configurable": {"thread_id": connection["thread_id"]},
    }

    try:
        # The checkpointer returns the FULL message history for this
        # thread_id (every prior turn), so grab its length first — that's
        # how we slice out just this turn's new messages below.
        prior_state = agent.get_state(run_config)
        prior_len = len(prior_state.values.get("messages", []))

        # invoke() runs the agent's ReAct loop to completion: model -> tool
        # calls -> tool results -> model -> ... until the model replies
        # without calling a tool. final_state["messages"] is the whole
        # conversation so far, old turns included.
        final_state = agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=run_config,
        )
        trajectory = final_state.get("messages", [])[prior_len:]
    except Exception as e:
        return f"Error: {e}"

    # The agent only stops once the model replies with no tool calls, so the
    # last message in this turn's trajectory is always the answer.
    answer = trajectory[-1].content if trajectory else None
    steps = trajectory[:-1] if answer else trajectory
    steps = [msg for msg in steps if msg.type != "human"]

    if not steps:
        return answer or "The agent didn't return an answer."

    # One parent "Agent trajectory" thought holds every step as a nested
    # child (via parent_id) — a single collapsible instead of one per step.
    parent_id = str(uuid.uuid4())
    trajectory_thought = {
        "role": "assistant",
        "content": "",
        "metadata": {"title": "🔍 Agent trajectory", "id": parent_id, "status": "done"},
    }
    thoughts = [describe_step(msg, parent_id) for msg in steps]
    return [trajectory_thought, *thoughts, answer or "The agent didn't return an answer."]


with gr.Blocks(title="SQL Deep Agent", fill_height=True) as demo:
    gr.Markdown("# SQL Deep Agent")

    # connection holds {"agent", "tool_names", "tables", "model", "thread_id"}
    # or None — Gradio's gr.State is per-browser-session, like Streamlit's
    # session_state, so each visitor gets their own agent connection.
    connection = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Configuration")
            model = gr.Dropdown(
                ["qwen3.5:4b-mlx", "gemma4:e4b", "nemotron-3-nano:4b"],
                value="qwen3.5:4b-mlx",
                label="Model",
            )
            base_url = gr.Textbox(value="http://localhost:11434", label="Ollama base URL")
            gr.Markdown("### Database")
            db_uri = gr.Textbox(
                value="sqlite:///data/Chinook.db",
                label="SQLAlchemy URI",
                info="e.g. sqlite:///data/Chinook.db, postgresql://user:pw@host/db",
            )
            max_steps = gr.Slider(5, 100, value=50, step=5, label="Max recursion steps")

            with gr.Row():
                connect_btn = gr.Button("Connect", variant="primary")
                clear_btn = gr.Button("Clear chat")

            status = gr.Markdown("")
            with gr.Accordion("Tools (0)", open=False) as tools_accordion:
                tools_md = gr.Markdown("")
            with gr.Accordion("Tables (0)", open=False) as tables_accordion:
                tables_md = gr.Markdown("")

        with gr.Column(scale=3):
            chat_interface = gr.ChatInterface(
                respond,
                additional_inputs=[connection, max_steps],
                chatbot=gr.Chatbot(height="85vh"),
            )

    connect_btn.click(
        connect,
        inputs=[model, base_url, db_uri],
        outputs=[connection, status, tools_accordion, tools_md, tables_accordion, tables_md],
    )

    clear_btn.click(clear_chat, inputs=[connection], outputs=[connection, chat_interface.chatbot])


if __name__ == "__main__":
    demo.launch()
