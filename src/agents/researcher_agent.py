import logging
from typing import Any, Dict

from src.agents.langgraph_agents import (
    LLMClient,
    run_web_search_from_config,
    run_paper_search_from_config,
)


def make_researcher_node(
    config: Dict[str, Any],
    llm: LLMClient,
    researcher_prompt: str,
    logger: logging.Logger | None = None,
):
    max_evidence_chars = (
        config.get("agents", {}).get("researcher", {}).get("max_evidence_chars", 1500)
    )

    def _researcher(state: Dict[str, Any]) -> Dict[str, Any]:
        if logger:
            logger.info("Entering node: Researcher")
        query = state["query"]
        plan = state.get("plan", "")
        web_results = run_web_search_from_config(config, query)
        paper_results = run_paper_search_from_config(config, query)

        findings_context = ""
        if web_results:
            findings_context += f"Web search results:\n{web_results}\n\n"
        if paper_results:
            findings_context += f"Paper search results:\n{paper_results}\n\n"
        if not findings_context:
            findings_context = "No tool results were available; provide best-effort desk research."

        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Research plan:\n{plan}\n\n"
            f"Findings to summarize:\n{findings_context}\n\n"
            "Provide concise evidence bullets with source tags."
        )
        try:
            evidence = llm.chat(researcher_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            evidence = f"[researcher error] {exc}"
            if logger:
                logger.error("Researcher failed: %s", exc, exc_info=True)
        if logger:
            logger.info("Researcher reply (len=%d)", len(str(evidence)))
        if isinstance(evidence, str) and len(evidence) > max_evidence_chars:
            if logger:
                logger.info(
                    "Researcher evidence trimmed from %d to %d chars",
                    len(evidence),
                    max_evidence_chars,
                )
            evidence = evidence[:max_evidence_chars] + "\n[trimmed]"
        conversation = state.get("conversation", []) + [
            {"source": "Researcher", "content": evidence}
        ]
        return {"evidence": evidence, "conversation": conversation}

    return _researcher
