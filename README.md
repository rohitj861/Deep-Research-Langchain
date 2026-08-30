# Deep Research AI

A research agent built on **[`deepagents`](https://github.com/langchain-ai/deepagents)** — LangChain's
implementation of the deep agent pattern. Ask a question, and the agent plans a
research program, delegates each sub-investigation to a subagent with its own clean
context window, writes notes to a virtual filesystem, drafts a report, has it critiqued,
and revises it.

The whole thing is one `create_deep_agent` call.

## What "deep agent" means here

A shallow agent is a while-loop over tool calls; it loses the plot on long tasks because
everything competes for one context window. A deep agent adds four things, all of which
`create_deep_agent` wires up for you:

| Capability | Tool | What it replaces in the old version of this repo |
| --- | --- | --- |
| **Planning** | `write_todos` | the hand-written `planner.py` |
| **Delegation / context isolation** | `task` → subagents | the hand-written `executor.py` |
| **Context offloading** | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | passing giant strings between functions |
| **A detailed system prompt** | `system_prompt` | `prompts.py`, now written for one agent instead of four calls |

The previous code called an LLM four times in a fixed sequence. The agent now decides how
many sub-investigations it needs, reads its own notes before choosing the next question,
and revises the report after a critique pass — with the long text living in files rather
than in the message history.

## Architecture

```
                       ┌──────────────────────────────┐
   your question ─────▶│      orchestrator agent      │
                       │  write_todos · task · files  │
                       └───────┬──────────────┬───────┘
                     task      │              │      task
              ┌───────────────▼──┐        ┌──▼───────────────┐
              │  research-agent  │  ...   │  critique-agent  │
              │  tavily_search   │        │  reads the notes │
              │  own context     │        │  own context     │
              └────────┬─────────┘        └─────────┬────────┘
                       │ write_file                 │ findings
                       ▼                            ▼
              /notes/*.md  ────────────▶  /final_report.md  ──▶ Markdown / PDF
```

Each `task` call runs a subagent in a **fresh context window**. It sees only the
instruction it was handed, does its research, writes a note file, and returns a short
summary. The orchestrator's context stays small no matter how much reading happened.

## Project structure

```text
Deep-Research-Langchain/
├── agent.py              # create_deep_agent(...) — the core of the project
├── prompts.py            # orchestrator + subagent system prompts
├── config.py             # providers as "provider:model" specs, depth presets
├── app.py                # Streamlit UI (streams tools, todos, files, report)
├── ui.py                 # presentation helpers, importable without running app.py
├── auth.py               # shared-password gate for the deployed app
├── exporter.py           # markdown -> PDF (headings, lists, tables)
├── errors.py             # provider errors -> one actionable line
├── tools/
│   ├── search.py         # Tavily web search for the research subagent
│   └── weather.py        # get_weather — worked example of a custom tool
├── utils/
│   └── files.py          # read the agent's virtual filesystem
├── examples/
│   ├── quickstart.py     # the smallest possible deep agent
│   └── research_cli.py   # full research run from the terminal
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # then fill in your keys
streamlit run app.py
```

### Keys

| Variable | Needed for |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI — the default provider. Paid only; no free tier. |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Gemini — the alternative provider. Has a free tier. |
| `TAVILY_API_KEY` | live web search. **Optional but strongly recommended** — without it the agent researches from model knowledge only and says so in its notes. |

You need a key for **one** model provider, not both — whichever you select in the
sidebar. `TAVILY_API_KEY` is separate and applies either way: the model does the
reasoning, Tavily does the looking-up. Swapping the model provider does not change
how search works.

## The minimal version

`examples/quickstart.py` is the shape everything else builds on:

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Pune?"}]})
```

Even this bare call already has the filesystem and `task` tools. `agent.py` adds the
planning middleware, the research prompts, and the two subagents.

## Providers

Models are LangChain model specs — `"<provider>:<model>"` — resolved by `init_chat_model`,
so switching providers is a string change, not a new client class:

| Sidebar option | Spec | Env override | Cost |
| --- | --- | --- | --- |
| OpenAI (default) | `openai:gpt-5.4-mini` | `OPENAI_MODEL` | paid only |
| Gemini | `google_genai:gemini-3.1-flash-lite` | `GEMINI_MODEL` | free tier available |

Adding another provider back is about five lines: append an entry to `Provider` and
`PROVIDER_SPECS` in `config.py`, and install that `langchain-*` package. Anything
`init_chat_model` supports works — nothing else in the codebase needs to change.

### Watch the free-tier quota

One research run is a burst of model calls — roughly 10-25 at Basic depth, more at
Advanced. Some models have very small free allowances (`gemini-3.6-flash` permits 20
requests **per day**), which a single run will exhaust. If you see a quota error, switch
`GEMINI_MODEL` to a model with more headroom, or enable billing on your provider account.

On OpenAI this shows up as cost rather than a quota wall — every run bills, and Advanced
depth multiplies it. Start on Basic when trying a new model.

The model client is rate limited (`REQUESTS_PER_MINUTE`, default 12) and retries
transient failures (`MAX_RETRIES`) so per-minute caps do not kill a run halfway through.
Lower `REQUESTS_PER_MINUTE` if you still hit 429s.

## Deploying to Streamlit Community Cloud

The app reads settings from the environment first and **Streamlit secrets** second, so
the same code runs locally off `.env` and on Cloud off the secrets panel — no branching.

1. **Push to a GitHub repo you own.** Cloud deploys from your repositories, so a clone
   still pointing at someone else's remote will not work.
2. At [share.streamlit.io](https://share.streamlit.io) choose **New app**, pick the repo
   and branch, and set **Main file path** to `app.py`.
3. Under **Advanced settings**, set the Python version to **3.11, 3.12, or 3.13**.
   `deepagents` requires 3.11+, so the older defaults will fail to install.
4. Paste your keys into **Secrets** (Advanced settings now, or Manage app → Settings →
   Secrets later). Use `.streamlit/secrets.toml.example` as the template:

   ```toml
   OPENAI_API_KEY = "sk-proj-..."
   TAVILY_API_KEY = "tvly-..."
   APP_PASSWORD = "a-long-random-string"
   DEFAULT_PROVIDER = "openai"
   ```

5. Deploy. Saving secrets reboots the app automatically.

### Notes

- **Never commit `.streamlit/secrets.toml`** — it is gitignored. Only
  `secrets.toml.example` (placeholders) and `config.toml` (theme, no secrets) are tracked.
- **Secrets do not reach `os.environ`.** Streamlit does not export them, so `config.py`
  publishes provider keys via `ensure_provider_env` and the Tavily key via
  `ensure_search_env` before the SDKs are constructed. Add a new key-reading SDK and you
  must do the same.
- **Set `APP_PASSWORD`.** Cloud apps are public by URL. With the password set, visitors
  see only a login form; without it the app runs but shows a warning banner. This is a
  shared front door, not per-user auth — for named access, use the viewer allowlist under
  Settings → Sharing.
- **Cost lives on the server.** A Cloud app spends *your* API budget for every visitor.
  Keep `RESEARCH_DEPTH=Basic` on a public deployment.
- **State is per-session and in memory.** Notes and reports vanish when the app sleeps or
  restarts; nothing is written to durable storage.

## Research depth

Depth sets how many sub-investigations the prompt asks for and the LangGraph step budget
for the run:

| Depth | Sub-investigations | Searches per task | Recursion limit |
| --- | --- | --- | --- |
| Basic | 2 | 2 | 60 |
| Standard | 4 | 3 | 120 |
| Advanced | 6 | 5 | 250 |

## Running headless

```bash
python examples/research_cli.py "Compare vector databases for RAG in 2026"
```

Writes `research_report.md` and `research_report.pdf`.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The suite covers provider/model resolution, agent assembly (which tools end up bound,
which are deliberately withheld), the virtual-filesystem helpers, tool degradation
without a Tavily key, and the markdown-to-PDF renderer. Nothing in it hits a network.

## Notes on design

- **Shell execution is withheld on purpose.** `create_deep_agent` would expose `execute`
  and `delete`; `agent.py` passes an explicit `FilesystemMiddleware(tools=[...])` list
  that omits both. The filesystem here is virtual state, not your disk.
- **The orchestrator does not search.** Search belongs to `research-agent` so that raw
  search results never enter the orchestrator's context.
- **Every question produces a report.** The orchestrator prompt makes the workflow
  mandatory even for questions the model could answer off the top of its head, and
  `ensure_report` in `agent.py` is the backstop: if a turn somehow ends without
  `final_report.md`, it spends one more turn on the same thread writing it from the notes
  already gathered. A narrow question gets a shorter report, never no report.
- **Threads are checkpointed in memory**, so follow-up questions in the same Streamlit
  session keep the notes and report from the previous turn. "New session" clears them.
- **Provider errors are translated** (`errors.py`) into a headline plus a next step, so a
  quota or bad-key failure shows one actionable line instead of a page of API JSON.
