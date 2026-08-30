"""Deep agent assembly.

The whole planner -> executor -> report pipeline that this project used to hand-roll
is now expressed as one `create_deep_agent` call:

- **Planning** comes from `TodoListMiddleware` (the `write_todos` tool).
- **Execution** comes from the built-in `task` tool dispatching to `research-agent`,
  each call running in its own isolated context window.
- **Report generation** is the orchestrator writing `final_report.md` into the
  agent's virtual filesystem, then revising it after `critique-agent` reviews it.
"""

from typing import Any

from deepagents import FilesystemMiddleware, SubAgent, create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from config import (
    DEFAULT_DEPTH,
    REPORT_FILE,
    DEFAULT_PROVIDER,
    MAX_RETRIES,
    REQUESTS_PER_MINUTE,
    Provider,
    ensure_provider_env,
    get_depth_preset,
    get_model_spec,
)
from prompts import critique_subagent_prompt, orchestrator_prompt, research_subagent_prompt
from tools import build_search_tools, get_weather
from utils import read_file

NUDGE = (
    f"You ended that turn without writing `{REPORT_FILE}`. Write it now with `write_file`, "
    "following the report format in your instructions and using whatever notes and findings "
    "you already have. Reply with a short summary once the file exists."
)

# The virtual filesystem is where research notes live, so the agent keeps read/write
# but not shell execution. `execute` and `delete` are dropped deliberately.
FILESYSTEM_TOOLS = ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]


def resolve_model(provider: Provider | str) -> BaseChatModel:
    """Build the chat model for a provider, paced so free tiers survive a full run.

    A research run is a burst of model calls (orchestrator + every subagent turn).
    Unpaced, that trips per-minute quotas partway through and loses the work done
    so far, so the client is rate limited and retries transient failures.
    """
    ensure_provider_env(provider)
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=max(REQUESTS_PER_MINUTE, 1) / 60,
        check_every_n_seconds=0.5,
        max_bucket_size=2,
    )
    return init_chat_model(get_model_spec(provider), rate_limiter=rate_limiter, max_retries=MAX_RETRIES)


def build_subagents(searches_per_task: int, research_tools: list[BaseTool]) -> list[SubAgent]:
    """Specialists the orchestrator can delegate to via the `task` tool."""
    research_agent: SubAgent = {
        "name": "research-agent",
        "description": (
            "Researches ONE narrow question end to end and writes its findings to a file. "
            "Give it a single self-contained question plus the note path to write to. "
            "Call it once per sub-topic; it cannot see the main conversation."
        ),
        "system_prompt": research_subagent_prompt(searches_per_task),
        "tools": research_tools,
    }

    critique_agent: SubAgent = {
        "name": "critique-agent",
        "description": (
            "Reviews the drafted final_report.md against the research notes and returns a "
            "prioritized list of problems to fix. Call it after the report is written."
        ),
        "system_prompt": critique_subagent_prompt(),
        "tools": [],
    }

    return [research_agent, critique_agent]


def build_research_agent(
    provider: Provider | str = DEFAULT_PROVIDER,
    depth: str = DEFAULT_DEPTH,
    *,
    model: BaseChatModel | None = None,
    checkpointer: Any | None = None,
    extra_tools: list[BaseTool] | None = None,
) -> CompiledStateGraph:
    """Build the Deep Research agent for a provider and research depth.

    Args:
        provider: Which entry in `config.Provider` to use.
        depth: A key of `config.DEPTH_PRESETS`.
        model: A pre-built chat model, used instead of resolving `provider`.
            Handy for tests and for models `init_chat_model` cannot construct.
        checkpointer: LangGraph checkpointer. Defaults to in-memory, which gives
            follow-up turns access to the previous turn's notes and report.
        extra_tools: Additional tools handed to both the orchestrator and the
            research subagent.
    """
    preset = get_depth_preset(depth)

    search_tools = build_search_tools()
    research_tools: list[BaseTool] = [*search_tools, *(extra_tools or [])]

    return create_deep_agent(
        model=model or resolve_model(provider),
        # The orchestrator delegates searching; it keeps only the demo tool plus
        # whatever the caller adds.
        tools=[get_weather, *(extra_tools or [])],
        system_prompt=orchestrator_prompt(preset.subagent_calls),
        subagents=build_subagents(preset.searches_per_task, research_tools),
        middleware=[
            TodoListMiddleware(),
            FilesystemMiddleware(tools=FILESYSTEM_TOOLS),
        ],
        checkpointer=checkpointer or InMemorySaver(),
    )


def ensure_report(agent: CompiledStateGraph, config: dict, state: dict) -> dict:
    """Guarantee the run produced a report.

    The system prompt makes the workflow mandatory, but a model can still shortcut it
    on a question it judges trivial. One bounded follow-up turn on the same thread
    covers that case; the agent keeps its notes, so this is a write, not a re-research.
    Returns the state after the nudge, or the original state if a report was there.
    """
    if read_file(state.get("files", {}), REPORT_FILE):
        return state

    agent.invoke({"messages": [{"role": "user", "content": NUDGE}]}, config)
    return agent.get_state(config).values


def run_config(depth: str = DEFAULT_DEPTH, thread_id: str = "default") -> dict:
    """LangGraph config: step budget for the depth, plus the conversation thread."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": get_depth_preset(depth).recursion_limit,
    }
