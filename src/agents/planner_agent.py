import logging
from typing import Any, Dict

from src.agents.langgraph_agents import LLMClient


def make_planner_node(llm: LLMClient, planner_prompt: str, logger: logging.Logger | None = None):
    def _planner(state: Dict[str, Any]) -> Dict[str, Any]:
        if logger:
            logger.info("Entering node: Planner")
        query = state["query"]
        user_prompt = f"User query:\n{query}\n\nCreate the plan now."
        try:
            plan = llm.chat(planner_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            plan = f"[planner error] {exc}"
            if logger:
                logger.error("Planner failed: %s", exc, exc_info=True)
        if logger:
            logger.info("Planner reply: %s", plan.replace("\n", " ")[:500])
        conversation = state.get("conversation", []) + [
            {"source": "Planner", "content": plan}
        ]
        return {"plan": plan, "conversation": conversation}

    return _planner
