"""SQL agent with an explicit ReAct loop (LangChain + Ollama) — Streamlit chat app.

Unlike `01_deep_agent_streamlit.py`, this version doesn't use `deepagents.create_deep_agent`
or a LangGraph checkpointer. The tool-calling loop and the conversation history are both
plain Python, so every step is visible here instead of inside a library.

Run with:
    uv add streamlit langchain-ollama langchain-community mlflow ddgs
    uv run streamlit run 02_deep_agent_detail_streamlit.py

View traces with:
    uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db
"""

# --- Setup ---------------------------------------------------------------
import inspect
import uuid

import mlflow
import streamlit as st

st.set_page_config(page_title="SQL Agent — Explicit ReAct (LangChain + Ollama)", layout="wide")

# One call enables tracing for every LangChain call made below. On its own,
# autolog starts a new top-level trace at every `llm_with_tools.invoke()` —
# there's no LangGraph Runnable here to nest them under, unlike 01_deep_agent.
# The @mlflow.trace on DeepAgent.invoke() below gives each turn's whole loop
# one parent span instead, so every model call in it nests under one trace.
# View them with `mlflow ui --backend-store-uri sqlite:///data/mlflow.db`.
mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
mlflow.set_experiment("02_deep_agent_detail")
mlflow.langchain.autolog()

# --- System prompt: steers the model inside the hand-rolled loop below. ----
INSTRUCTIONS = """You are an agent designed to interact with a SQL database.

Given an input question, create a syntactically correct sqlite query, run it,
inspect the results, and return a concise answer. Unless the user asks for a
specific number of rows, limit results to at most 10. Never `SELECT *` — only
request the columns you need.

Do NOT issue any DML statements (INSERT, UPDATE, DELETE, DROP, etc.).

Always verify table names with `sql_db_list_tables` and inspect schemas with
`sql_db_schema` before writing a query. Use `sql_db_query_checker` on any
non-trivial query before running it with `sql_db_query`.

Always try the SQL tools first. If the data isn't in the database — either
the question is about something outside it (e.g. background on an artist)
or a query comes back empty — use `duckduckgo_search` to look it up on the
web instead of saying you don't know.

Answer in plain language. Do not describe the database schema or the steps
you took.
"""

# --- Core LangChain pieces: model, tools, and the messages themselves ------
# No `deepagents`, no `langgraph`, and — unlike 01_deep_agent — no
# `SQLDatabaseToolkit` either. The tools below are plain Python, so there's
# nothing left hidden inside a toolkit class: this IS the toolkit.
from ddgs import DDGS
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama


# ----------------------------------------------------------------------------
# SQL Tools
# ----------------------------------------------------------------------------

class SQLTools:
    """The four SQL tools, as bound methods instead of four separate closures —
    `db`/`llm` live once on `self` rather than being captured per-function.
    `bind_tools()` reads a bound method exactly like a free function: `self`
    was already filled in when `self.sql_db_schema` was looked up, so it never
    shows up as a parameter in the generated schema."""

    def __init__(self, db: SQLDatabase, llm: ChatOllama):
        self.db = db
        self.llm = llm  # sql_db_query_checker needs a model to ask, not just the db

    def sql_db_list_tables(self) -> str:
        """List every usable table in the connected database.

        Takes no arguments. Call this first, before sql_db_schema, so you
        only ever ask about tables that actually exist.
        """
        return ", ".join(self.db.get_usable_table_names())

    def sql_db_schema(self, table_names: str) -> str:
        """Get the CREATE TABLE statement and a few sample rows for one or more tables.

        Args:
            table_names: A comma-separated list of table names, e.g. "Artist, Album".
                Only pass names returned by sql_db_list_tables — made-up names
                raise an error here instead of in sql_db_query.
        """
        names = [name.strip() for name in table_names.split(",")]
        return self.db.get_table_info_no_throw(names)

    def sql_db_query_checker(self, query: str) -> str:
        """Double-check a SQL query for common mistakes before running it.

        Args:
            query: The SQL query to check.

        Always call this before sql_db_query. It asks the model itself (a
        second, tool-free call to `self.llm`) to look for things like NOT IN
        with NULLs, UNION vs UNION ALL, exclusive BETWEEN ranges, or wrong
        join columns, and rewrite the query if it finds one — or return it
        unchanged if it looks fine.
        """
        prompt = (
            f"{query}\n"
            f"Double check the {self.db.dialect} query above for common mistakes, including:\n"
            "- Using NOT IN with NULL values\n"
            "- Using UNION when UNION ALL should have been used\n"
            "- Using BETWEEN for exclusive ranges\n"
            "- Data type mismatch in predicates\n"
            "- Properly quoting identifiers\n"
            "- Using the correct number of arguments for functions\n"
            "- Casting to the correct data type\n"
            "- Using the proper columns for joins\n\n"
            "If there are any of the above mistakes, rewrite the query. If "
            "there are no mistakes, just reproduce the original query.\n\n"
            "Output the final SQL query only.\n\nSQL Query:"
        )
        # Note: `self.llm`, not a tool-bound model — this call doesn't need
        # (or want) the model to ask for another tool, just to answer in text.
        return self.llm.invoke(prompt).content

    def sql_db_query(self, query: str) -> str:
        """Run a SQL query against the database and return the matching rows.

        Args:
            query: A syntactically correct SQL query. Never SELECT * — only
                request the columns you need, and never issue DML (INSERT,
                UPDATE, DELETE, DROP, etc.).

        If the query is invalid, the database's own error message comes back
        as the result instead of raising — rewrite the query and try again
        rather than giving up.
        """
        return self.db.run_no_throw(query)

    def get_tools(self) -> list:
        """The bound methods bind_tools() (and the dispatch loop) should see."""
        return [self.sql_db_list_tables, self.sql_db_schema, self.sql_db_query_checker, self.sql_db_query]


# ----------------------------------------------------------------------------
# Search Tool
# ----------------------------------------------------------------------------

def duckduckgo_search(query: str) -> str:
    """Search the web for something the database can't answer.

    Args:
        query: What to search for, e.g. an artist's background or a current
            event — not SQL.

    Calls the `ddgs` package directly (the same library DuckDuckGoSearchRun
    wraps) instead of going through a LangChain tool class.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No good DuckDuckGo Search Result was found"
    return " ".join(result["body"] for result in results)


# ----------------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------------

class InMemorySaver:
    """A minimal, hand-rolled stand-in for `langgraph.checkpoint.memory.InMemorySaver`:
    a plain dict keyed by `thread_id`, each holding that conversation's entire
    message list. No persistence beyond the process's own memory — same as the
    real one — just without any of the LangGraph checkpoint machinery around it.
    This is what actually "remembers" a conversation between invoke() calls."""

    def __init__(self):
        self._threads: dict[str, list] = {}

    def get_messages(self, thread_id: str) -> list:
        """Return `thread_id`'s message history, creating a new empty one the
        first time it's seen."""
        return self._threads.setdefault(thread_id, [])

    def append(self, thread_id: str, message) -> None:
        """Add one message to the end of `thread_id`'s history — the only way
        callers should mutate it, rather than appending to `get_messages()`'s
        return value themselves."""
        self.get_messages(thread_id).append(message)

    def seed_system_message(self, thread_id: str, system_prompt: str) -> None:
        """Make sure `thread_id` starts with the system prompt — a no-op if
        it already has any history. Called once per thread, the same way
        create_deep_agent's system_prompt is baked into every run rather than
        being the caller's job."""
        if not self.get_messages(thread_id):
            self.append(thread_id, SystemMessage(content=system_prompt))


# ----------------------------------------------------------------------------
# Model Tools
# ----------------------------------------------------------------------------

class ModelTools:
    """Builds OpenAI-style function-calling schemas for plain Python tools —
    the one real piece of bind_tools() that needs reimplementing. The rest
    of it (`model.bind(tools=schemas)`) turned out to be nothing more than a
    `RunnableBinding` that replays `tools=schemas` on every `.invoke()` call
    (confirmed by reading `Runnable.bind`/`RunnableBinding.invoke`) — so
    DeepAgent just passes `tools=` explicitly instead of going through it."""

    @staticmethod
    def schemas(tools: list) -> list[dict]:
        """One {"type": "function", "function": {...}} schema per tool —
        exactly the list `model.bind_tools(tools)` would have built before
        attaching it to a RunnableBinding."""
        return [ModelTools._tool_schema(tool) for tool in tools]

    @staticmethod
    def _tool_schema(tool) -> dict:
        """Build one tool's {"type": "function", "function": {...}} schema by
        reading its own name, docstring, and signature — the same
        information a LangChain BaseTool's .name/.description/.args_schema
        would otherwise carry, reconstructed from a plain function instead.

        For example, calling this on SQLTools.sql_db_schema — whose docstring
        is "Get the CREATE TABLE statement and a few sample rows for one or
        more tables.\n\nArgs:\n    table_names: A comma-separated list of
        table names..." — produces exactly this (verified by actually
        running it):

            {
                "type": "function",
                "function": {
                    "name": "sql_db_schema",
                    "description": "Get the CREATE TABLE statement and a few sample rows for one or more tables.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_names": {
                                "type": "string",
                                "description": "A comma-separated list of table names, e.g. \"Artist, Album\". Only pass names returned by sql_db_list_tables — made-up names raise an error here instead of in sql_db_query."
                            }
                        },
                        "required": ["table_names"]
                    }
                }
            }

        This is the exact shape Ollama's chat API expects in its "tools"
        list, and the exact shape `convert_to_openai_tool` (what the real
        bind_tools() calls internally) produces for the same function.
        """
        doc = inspect.getdoc(tool) or ""
        # Split the docstring in two at the "Args:" heading: everything
        # before it is the tool's top-level description (what the model
        # reads to decide *when* to call this tool at all, before it even
        # gets to arguments); everything after is parsed below into one
        # description per parameter.
        description, _, args_doc = doc.partition("Args:")
        description = " ".join(description.split())  # collapse wrapped lines into one

        # Google-style "name: description..." lines under "Args:". Multi-line
        # descriptions (like table_names's above, which wraps onto a second
        # line) don't start with "name:", so they're folded into whichever
        # argument's description was being built when they were seen.
        arg_descriptions: dict[str, str] = {}
        last_arg = None
        for line in args_doc.splitlines():
            line = line.strip()
            if not line:
                continue
            name, sep, rest = line.partition(":")
            if sep and name.isidentifier():
                arg_descriptions[name] = rest.strip()
                last_arg = name
            elif last_arg:
                arg_descriptions[last_arg] += " " + line

        # inspect.signature() is what finds the parameter *names* themselves
        # (arg_descriptions above only has their descriptions, parsed from
        # free text) — every one becomes a required string property, true
        # for all the tools in this file (SQL queries and search terms are
        # all plain text), so there's no need to handle other types here.
        properties = {}
        required = []
        for name, param in inspect.signature(tool).parameters.items():
            if name == "self":  # bound methods never show it, but be safe
                continue
            properties[name] = {"type": "string", "description": arg_descriptions.get(name, "")}
            if param.default is inspect.Parameter.empty:
                required.append(name)

        # sql_db_list_tables takes no arguments at all, so `properties` stays
        # {} and `required` stays [] (omitted below) for that one — Ollama
        # reads that as "call this with an empty args object".
        parameters = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required

        return {
            "type": "function",
            "function": {"name": tool.__name__, "description": description, "parameters": parameters},
        }


# ----------------------------------------------------------------------------
# Agent harness
# ----------------------------------------------------------------------------

class DeepAgent:
    """A tool-bound model plus the loop that drives it — everything
    `create_deep_agent` would otherwise hide. `.invoke()` is deliberately the
    same shape as 01_deep_agent's LangGraph agent (`agent.invoke(messages,
    config={"configurable": {"thread_id": ...}})`), just with the ReAct loop
    and the message history both written out instead of compiled into a graph."""

    def __init__(self, model: ChatOllama, tools: list, system_prompt: str,
                 checkpointer: InMemorySaver):
        # No RunnableBinding needed: ModelTools.schemas() builds the JSON
        # schema list once, here — the same {"name", "description",
        # "parameters"} conversion model.bind_tools() would have done — and
        # invoke() below just passes tools=self.tool_schemas on every call,
        # which is all that wrapper was ever doing under the hood anyway.
        self.llm = model
        self.tool_schemas = ModelTools.schemas(tools)
        self.tools_by_name = {tool.__name__: tool for tool in tools}
        self.tool_names = list(self.tools_by_name)
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer

    @mlflow.trace(name="DeepAgent.invoke", span_type="AGENT")
    def invoke(self, question: str, thread_id: str, max_steps: int) -> str | None:
        """Ask the model, and if it asked to call a tool instead of answering,
        run that tool and feed the result back as a ToolMessage — then ask
        again. Repeat until it replies with no tool calls, or max_steps is
        reached. The checkpointer — not this method — is what holds the
        conversation's memory: every read and write to it goes through
        `self.checkpointer`, with `thread_id` just a key into it, the same
        role it plays for 01_deep_agent's LangGraph checkpointer."""
        self.checkpointer.seed_system_message(thread_id, self.system_prompt)
        self.checkpointer.append(thread_id, HumanMessage(content=question))

        for _ in range(max_steps):
            # Get list of messages so far.
            messages = self.checkpointer.get_messages(thread_id)

            # One "thought" of the ReAct loop: hand the whole conversation so
            # far to the model — plus the tool schemas, passed explicitly on
            # this call instead of through a pre-bound wrapper — and let it
            # decide what comes next. `messages` already reflects every
            # earlier step in this turn (and every prior turn on this
            # thread_id), since each append above mutates the checkpointer's
            # own list in place.
            ai_message = self.llm.invoke(messages, tools=self.tool_schemas)
            self.checkpointer.append(thread_id, ai_message)

            # No tool_calls means the model chose to answer in plain text
            # instead of acting — that's the loop's only exit besides running
            # out of max_steps, so the turn ends here with its final answer.
            if not ai_message.tool_calls:
                return ai_message.content

            # A single model turn can ask for several tool calls at once —
            # run each one and feed every result back before asking again.
            for tool_call in ai_message.tool_calls:
                tool = self.tools_by_name[tool_call["name"]]
                result = tool(**tool_call["args"])
                self.checkpointer.append(thread_id, 
                                         ToolMessage(content=str(result), 
                                                     tool_call_id=tool_call["id"]))

        return None  # ran out of steps without a final answer


def build_agent(model: str, base_url: str, db_uri: str):
    """Configure the model + tools and return (agent, table_names)."""

    # 1. Model — a local Ollama model. temperature=0 keeps SQL generation
    # deterministic; validate_model_on_init fails fast if it isn't pulled yet.
    llm = ChatOllama(model=model, base_url=base_url, 
                     validate_model_on_init=True, temperature=0)
    db = SQLDatabase.from_uri(db_uri)

    # 2. Tools — bind_tools() accepts plain functions and bound methods alike,
    # not just BaseTool objects: it reads each one's name, type hints, and
    # docstring to build the same {"name", "description", "parameters"}
    # schema a BaseTool would've carried in `.args_schema` — no class
    # required for that part. `query`/`table_names` below become the JSON
    # parameters; everything in the docstring above "Args:" becomes the
    # tool's top-level description (what the model uses to decide *when* to
    # call it).
    tools = SQLTools(db=db, llm=llm).get_tools() + [duckduckgo_search]

    # 3. Assembled: DeepAgent takes the same (model, tools, system_prompt,
    # checkpointer) shape `create_deep_agent` does in 01_deep_agent — binding
    # the tools to the model happens inside its __init__ (see DeepAgent
    # above), and a fresh InMemorySaver gives this connection its own
    # independent conversation memory. bind_tools() does NOT give the model
    # the ability to run anything — tools never execute there either. It
    # converts each function above into the JSON
    # schema OpenAI-style "function calling" APIs expect:
    #   {"type": "function", "function": {"name": ..., "description": ...,
    #     "parameters": <json-schema built from the function's signature>}}
    # and returns a wrapped model (`_ChatModelBinding`) that attaches that list
    # of schemas to every request it sends to Ollama's chat API from now on.
    # Ollama's model then does two things on its own, as part of generating
    # its reply: (a) decide whether this turn needs a tool at all, and (b) if
    # so, which one and with what arguments — picked by name/description, the
    # same way it picks any other words to generate. LangChain parses that
    # raw response back into an `AIMessage`; if the model asked for a tool,
    # `.tool_calls` comes back populated, e.g.:
    #   [{"name": "sql_db_list_tables", "args": {},
    #     "id": "<uuid>", "type": "tool_call"}]
    # and `.content` is usually empty. If it just answered in words instead,
    # `.tool_calls` is an empty list and `.content` holds the answer.
    # `DeepAgent.invoke()` is what reads `.tool_calls` and actually calls
    # `tools_by_name[name](**args)` — binding only shapes the request.
    agent = DeepAgent(model=llm, tools=tools, system_prompt=INSTRUCTIONS, 
                      checkpointer=InMemorySaver())
    return agent, db.get_usable_table_names()


def render_trajectory(trajectory):
    """Render an expander listing this turn's steps: tool calls (name + args)
    and any plain message content, in the order they happened."""
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

    max_steps = st.slider("Max tool-calling steps", 5, 100, 50)

    col_connect, col_clear = st.columns(2)
    connect_clicked = col_connect.button("Connect", type="primary", width="stretch")
    if col_clear.button("Clear chat", width="stretch"):
        # A fresh thread_id makes the checkpointer start a brand-new
        # conversation on the next turn — the old thread's history just
        # becomes unreachable inside the InMemorySaver, the same way it would
        # with 01_deep_agent's LangGraph checkpointer.
        st.session_state["messages"] = []
        st.session_state["thread_id"] = str(uuid.uuid4())

    # Once connected, show what the agent has access to.
    if st.session_state.get("agent_ready"):
        st.success(f"Connected — {st.session_state.get('connected_model')}")
        with st.expander(f"Tools ({len(st.session_state.get('tool_names', []))})"):
            st.write(st.session_state.get("tool_names", []))
        with st.expander(f"Tables ({len(st.session_state.get('tables', []))})"):
            st.write(st.session_state.get("tables", []))


# Build (or rebuild) the model + tools and stash them in session_state — this
# is the only place build_agent() is called, so changing a sidebar setting has
# no effect until you click Connect again.
if connect_clicked:
    try:
        with st.spinner("Connecting…"):
            agent, tables = build_agent(model, base_url, db_uri)
        st.session_state["agent"] = agent
        st.session_state["tool_names"] = agent.tool_names
        st.session_state["tables"] = tables
        st.session_state["connected_model"] = model
        st.session_state["agent_ready"] = True
        st.session_state.setdefault("messages", [])
        # thread_id is the checkpointer's conversation key — created once per
        # connection and reused across turns (see agent.invoke below) so the
        # agent remembers earlier questions in the same session.
        st.session_state.setdefault("thread_id", str(uuid.uuid4()))
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")


# --- Main: chat -------------------------------------------------------------
st.title("SQL Agent — Explicit ReAct Loop")
st.session_state.setdefault("messages", [])

render_history()

# st.chat_input() is only called (and so only rendered) inside this elif, so
# the input box itself doesn't appear until an agent is connected.
if not st.session_state.get("agent_ready"):
    st.info("Set the database URI in the sidebar, then click **Connect**.")
elif question := st.chat_input("Ask a question about the database…"):
    # Draw the user's own message immediately, before calling the (slow)
    # model below — render_history() already ran, so this turn's question
    # isn't in it yet.
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer = None
        agent = st.session_state["agent"]
        thread_id = st.session_state["thread_id"]
        # The checkpointer returns the FULL message history for this
        # thread_id (every prior turn), so grab its length first — that's
        # how we slice out just this turn's new messages below, the same way
        # 01_deep_agent slices its checkpointer's prior_state.
        prior_len = len(agent.checkpointer.get_messages(thread_id))

        with st.spinner("Thinking…"):
            try:
                answer = agent.invoke(question, thread_id, max_steps)
            except Exception as e:
                st.exception(e)

        # On a brand-new thread, invoke() seeds the system prompt before the
        # human question — which shifts everything after it, so slicing from
        # prior_len would otherwise start with that system message instead of
        # this turn's human question. Filtering it out keeps the trajectory
        # to just this turn's actual steps regardless of that one-time shift.
        full_history = agent.checkpointer.get_messages(thread_id)
        trajectory = [msg for msg in full_history[prior_len:] if msg.type != "system"]
        # Only render/save a reply if the loop actually produced one — it
        # stays empty if an exception was raised above, or None if max_steps
        # was hit with no final answer.
        if answer:
            st.markdown(answer)
            render_trajectory(trajectory)
            # Mirror what render_history() draws for a saved message, so this
            # turn looks identical to itself on the next rerun.
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "trajectory": trajectory}
            )
        elif answer is None and trajectory:
            st.warning(f"Hit the {max_steps}-step limit without a final answer.")
