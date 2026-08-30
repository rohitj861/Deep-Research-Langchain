"""Web search tool for the research subagents.

Backed by Tavily when ``TAVILY_API_KEY`` is set. Without a key the agent runs in
model-knowledge-only mode and the prompts tell it to flag that.
"""

from langchain_core.tools import BaseTool

from config import ensure_search_env, search_enabled

MAX_RESULTS = 5


def build_search_tools() -> list[BaseTool]:
    """Return the search tools available in this environment (possibly empty)."""
    if not search_enabled():
        return []

    # TavilySearch reads TAVILY_API_KEY from the environment, so a Cloud secret
    # has to be published there before the tool is constructed.
    ensure_search_env()

    from langchain_tavily import TavilySearch

    return [
        TavilySearch(
            max_results=MAX_RESULTS,
            topic="general",
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
        )
    ]
