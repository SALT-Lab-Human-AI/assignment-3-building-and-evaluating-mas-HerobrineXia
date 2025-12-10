import logging
from typing import Any, Dict

from src.agents.langgraph_agents import LLMClient


def make_critic_node(llm: LLMClient, critic_prompt: str, logger: logging.Logger | None = None):
    def _critic(state: Dict[str, Any]) -> Dict[str, Any]:
        if logger:
            logger.info("Entering node: Critic")
        query = state["query"]
        draft = state.get("draft", "")
        plan = state.get("plan", "")
        evidence = state.get("evidence", "")
        iteration = state.get("iteration", 0)
        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Draft to review:\n{draft}\n\n"
            "Review and decide if approved or needs revision. Keep feedback concise (e.g., a short bullet list)."
        )
        try:
            critique = llm.chat(critic_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            critique = f"[critic error] {exc}"
            if logger:
                logger.error("Critic failed: %s", exc, exc_info=True)
        next_iteration = iteration + 1 if "NEEDS REVISION" in critique else iteration
        if logger:
            logger.info("Critic reply: %s", critique.replace("\n", " ")[:500])
        conversation = state.get("conversation", []) + [
            {"source": "Critic", "content": critique}
        ]
        return {
            "critique": critique,
            "conversation": conversation,
            "iteration": next_iteration,
        }

    return _critic
