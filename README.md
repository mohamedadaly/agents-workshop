# Agents Workshop

A Streamlit chat app that answers questions about a SQL database using a
LangChain "deep agent" running on a local Ollama model, with MLflow tracing.

## Setup

### 1. Install Git (if you don't have it)

- **macOS**: `xcode-select --install` (or install via [Homebrew](https://brew.sh): `brew install git`)
- **Windows**: [git-scm.com/downloads](https://git-scm.com/downloads)
- **Linux**: `sudo apt install git` (Debian/Ubuntu) or your distro's package manager

### 2. Clone this repository

```bash
git clone <repo-url>
cd agents-workshop
```

### 3. Install an editor or terminal

- **VS Code** (recommended): download from [code.visualstudio.com](https://code.visualstudio.com), then open this folder with `code .`
- Or just use your system terminal — no editor required to run the app.

### 4. Install Ollama and pull a model

Download and install Ollama from [ollama.com/download](https://ollama.com/download).

Then pull the default model used by this app:

```bash
ollama pull qwen3.5:4b-mlx
```

Make sure Ollama is running (it starts automatically after install, or run `ollama serve`).

### 5. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(Windows: see [docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation/))

### 6. Install project dependencies

From inside the `agents-workshop` folder:

```bash
uv sync
```

This reads `pyproject.toml` / `uv.lock` and creates a local `.venv` with everything installed
(`streamlit`, `deepagents`, `langchain-ollama`, `langchain-community`, `mlflow`).

## Running the app

### 7. Start the Streamlit chat app

```bash
uv run streamlit run 01_deep_agent.py
```

This opens the app in your browser. In the sidebar, pick a model, confirm the database URI
(defaults to the included `Chinook.db`), and click **Connect** before chatting.

### 8. (Optional) View MLflow traces

Every question you ask is traced automatically. To browse the traces in a separate terminal:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the URL it prints (typically [http://localhost:5000](http://localhost:5000)).
