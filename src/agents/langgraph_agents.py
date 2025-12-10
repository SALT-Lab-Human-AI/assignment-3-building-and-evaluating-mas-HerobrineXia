"""
LangGraph agent utilities (prompts, LLM client, tool helpers).

This module centralizes agent-related helpers so the orchestrator remains thin.
"""

import logging
import os
import concurrent.futures
from typing import Any, Dict, Optional

from openai import OpenAI

from src.tools.web_search import web_search
from src.tools.paper_search import paper_search

logger = logging.getLogger("langgraph_agents")


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completion client."""

    def __init__(self, model_cfg: Dict[str, Any], max_tokens_override: Optional[int] = None):
        provider = model_cfg.get("provider", "openai")
        model_name = model_cfg.get("name", "gpt-4o-mini")
        max_tokens = model_cfg.get("max_tokens", 2048)

        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            base_url = "https://api.groq.com/openai/v1"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError(f"Missing API key for provider '{provider}'")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_tokens = max_tokens_override or max_tokens
        self.provider = provider

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.max_tokens,
        }
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed (model=%s): %s", self.model_name, exc, exc_info=True)
            raise

        choice = response.choices[0]
        content = getattr(choice.message, "content", "") or ""

        # If the model returned a tool/function call or empty text, log raw payload for debugging
        if not str(content).strip():
            logger.error(
                "LLM returned empty content (model=%s, finish_reason=%s). Raw choice=%r, response=%r",
                self.model_name,
                getattr(choice, "finish_reason", None),
                choice,
                response,
            )
            raise RuntimeError("LLM returned empty content.")

        return content


def load_prompts(config: Dict[str, Any]) -> Dict[str, str]:
    """Load agent prompts from config with safe defaults."""

    def get_prompt(agent_key: str, default: str) -> str:
        return (
            config.get("agents", {})
            .get(agent_key, {})
            .get("system_prompt", "")
            .strip()
            or default
        )

    return {
        "planner": get_prompt(
            "planner",
            """You are the Planner. Break the user query into a concise, actionable research plan.
List 3-6 focused steps, propose search strings, and end with 'PLAN COMPLETE'.""",
        ),
        "researcher": get_prompt(
            "researcher",
            """You are the Researcher. Use the provided tools summary.
Return concise bullet findings with inline tags like [Web1]/[Paper1] and end with 'RESEARCH COMPLETE'.""",
        ),
        "writer": get_prompt(
            "writer",
            """You are the Writer. Synthesize the findings into a concise, well-cited answer.
Include short intro, organized sections, inline tags, and a References list. End with 'DRAFT COMPLETE'.""",
        ),
        "critic": get_prompt(
            "critic",
            """You are the Critic. Check relevance, evidence, completeness, accuracy, and clarity.
If acceptable, reply with a brief approval including 'APPROVED - RESEARCH COMPLETE. TERMINATE'.
If not, list concise fixes and include 'NEEDS REVISION'.""",
        ),
    }


def run_web_search_from_config(config: Dict[str, Any], query: str) -> str:
    tool_cfg = config.get("tools", {}).get("web_search", {})
    if not tool_cfg.get("enabled", True):
        return ""
    try:
        provider = tool_cfg.get("provider", "tavily")
        max_results = tool_cfg.get("max_results", 5)

        # Always offload to a thread to avoid nested event loop issues
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(web_search, query, provider, max_results).result()
        if not str(result).strip():
            msg = f"[web_search] No results returned (provider={provider}). Check API key or network."
            logger.warning(msg)
            return msg
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Web search failed: {exc}")
        return f"[web_search error] {exc}"


def run_paper_search_from_config(config: Dict[str, Any], query: str) -> str:
    tool_cfg = config.get("tools", {}).get("paper_search", {})
    if not tool_cfg.get("enabled", False):
        return ""
    try:
        max_results = tool_cfg.get("max_results", 10)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(paper_search, query, max_results, None).result()
        if not str(result).strip():
            msg = "[paper_search] No results returned. Check API key or provider status."
            logger.warning(msg)
            return msg
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Paper search failed: {exc}")
        return f"[paper_search error] {exc}"
