import logging
from typing import Any, Dict

from src.agents.langgraph_agents import LLMClient


def make_writer_node(llm: LLMClient, writer_prompt: str, logger: logging.Logger | None = None):
    def _writer(state: Dict[str, Any]) -> Dict[str, Any]:
        if logger:
            logger.info("Entering node: Writer")
        query = state["query"]
        evidence = state.get("evidence", "")
        critique = state.get("critique", "")
        iteration = state.get("iteration", 0)
        previous_draft = state.get("draft", "")

        if "NEEDS REVISION" in critique and iteration > 0:
            # Revision path: use prior draft + latest critic feedback + research evidence
            user_prompt = (
                f"User query:\n{query}\n\n"
                f"Previous draft (keep only essential content, be concise):\n{previous_draft}\n\n"
                f"Research evidence (do not redo research, keep citations):\n{evidence}\n\n"
                f"Critic feedback to address (latest):\n{critique}\n\n"
                "Revise the draft above to address ONLY this feedback. Keep useful content, preserve citations, and produce an updated draft."
            )
        else:
            # Initial draft: use plan + evidence
            plan = state.get("plan", "")
            user_prompt = (
                f"User query:\n{query}\n\n"
                f"Research plan:\n{plan}\n\n"
                f"Findings from Researcher:\n{evidence}\n\n"
                "Write the draft now with citations and a brief References section."
            )
        try:
            draft = llm.chat(writer_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            draft = f"[writer error] {exc}"
            if logger:
                logger.error("Writer failed: %s", exc, exc_info=True)
        if logger:
            logger.info("Writer reply (len=%d)", len(str(draft)))
        conversation = state.get("conversation", []) + [
            {"source": "Writer", "content": draft}
        ]
        return {"draft": draft, "conversation": conversation}

    return _writer
