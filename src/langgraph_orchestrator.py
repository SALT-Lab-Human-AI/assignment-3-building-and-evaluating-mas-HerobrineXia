"""
LangGraph-Based Orchestrator

Workflow: Planner → Researcher → Writer → Critic (optional single revision loop back to Writer).
This replaces the previous AutoGen-based orchestrator.
"""

import logging
import re
from typing import Dict, Any, List, TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.langgraph_agents import LLMClient, load_prompts
from src.agents.planner_agent import make_planner_node
from src.agents.researcher_agent import make_researcher_node
from src.agents.writer_agent import make_writer_node
from src.agents.critic_agent import make_critic_node
from src.guardrails.safety_manager import SafetyManager


class ResearchState(TypedDict, total=False):
    """Shared state passed through the LangGraph workflow."""

    query: str
    plan: str
    evidence: str
    draft: str
    critique: str
    iteration: int
    conversation: List[Dict[str, str]]


class LangGraphOrchestrator:
    """
    Orchestrates multi-agent style research using LangGraph.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("langgraph_orchestrator")
        self._setup_logger()

        model_cfg = config.get("models", {}).get("default", {})
        # Allow writer to request a higher max token budget via override
        writer_cfg = config.get("agents", {}).get("writer", {})
        writer_max_tokens = writer_cfg.get("max_tokens")

        self.llm = LLMClient(model_cfg)
        # Separate client for writer if higher max_tokens requested
        self.writer_llm = (
            LLMClient(model_cfg, max_tokens_override=writer_max_tokens)
            if writer_max_tokens
            else self.llm
        )
        prompts = load_prompts(config)
        self.planner_prompt = prompts["planner"]
        self.researcher_prompt = prompts["researcher"]
        self.writer_prompt = prompts["writer"]
        self.critic_prompt = prompts["critic"]
        # Writer can use higher token budget; Critic kept concise via prompt only
        writer_cfg = config.get("agents", {}).get("writer", {})
        self.writer_max_tokens = writer_cfg.get("max_tokens", 4096)
        self.safety = SafetyManager(config.get("safety", {}))

        # Revision budget (how many times Critic can send work back to Writer)
        self.max_revisions = 3

    def _setup_logger(self):
        """Configure orchestrator logger using config logging settings."""
        log_cfg = self.config.get("logging", {})
        level_name = log_cfg.get("level", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        fmt = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file = log_cfg.get("file")

        self.logger.setLevel(level)
        if not self.logger.handlers:
            formatter = logging.Formatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            try:
                sh.setStream(open(1, "w", encoding="utf-8", closefd=False))  # type: ignore[arg-type]
            except Exception:
                pass
            self.logger.addHandler(sh)
            if log_file:
                import os

                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
        self.logger.propagate = False

    def _build_graph(self):
        graph = StateGraph(ResearchState)

        planner_node = make_planner_node(self.llm, self.planner_prompt, logger=self.logger)
        researcher_node = make_researcher_node(self.config, self.llm, self.researcher_prompt, logger=self.logger)
        writer_node = make_writer_node(self.writer_llm, self.writer_prompt, logger=self.logger)
        critic_node = make_critic_node(self.llm, self.critic_prompt, logger=self.logger)

        def route_from_critic(state: ResearchState) -> str:
            critique = state.get("critique", "")
            iteration = state.get("iteration", 0)
            if "NEEDS REVISION" in critique and iteration < self.max_revisions:
                return "revise"
            return "done"

        graph.add_node("planner", planner_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("writer", writer_node)
        graph.add_node("critic", critic_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "writer")
        graph.add_edge("writer", "critic")
        graph.add_conditional_edges(
            "critic",
            route_from_critic,
            {
                "revise": "writer",
                "done": END,
            },
        )

        return graph.compile()

    def process_query(self, query: str, max_rounds: int = 1) -> Dict[str, Any]:  # noqa: ARG002
        """Entry point used by CLI/Streamlit."""
        self.logger.info(f"Processing query: {query}")
        try:
            # Safety check on input
            input_check = self.safety.check_input_safety(query)
            if not input_check.get("safe", True):
                msg = self.config.get("safety", {}).get("on_violation", {}).get(
                    "message",
                    "I cannot process this request due to safety policies.",
                )
                return {
                    "query": query,
                    "error": "input_unsafe",
                    "response": msg,
                    "conversation_history": [],
                    "metadata": {
                        "error": True,
                        "safety_violations": input_check.get("violations", []),
                        "safety_events": self.safety.get_safety_stats(),
                    },
                }

            app = self._build_graph()
            initial_state: ResearchState = {
                "query": query,
                "plan": "",
                "evidence": "",
                "draft": "",
                "critique": "",
                "conversation": [],
                "iteration": 0,
                # placeholder keys if future nodes expect them; not used for trimming now
                "writer_max_chars": None,
                "critic_max_chars": None,
            }
            result: ResearchState = app.invoke(initial_state)
            final = self._extract_results(query, result)

            # Safety check on output
            output_check = self.safety.check_output_safety(
                final.get("response", ""), final.get("metadata", {}).get("research_findings")
            )
            if not output_check.get("safe", True):
                action = self.config.get("safety", {}).get("on_violation", {}).get("action", "refuse")
                message = self.config.get("safety", {}).get("on_violation", {}).get(
                    "message",
                    "I cannot process this request due to safety policies.",
                )
                if action == "refuse":
                    final["response"] = message
                final["metadata"]["safety_violations"] = output_check.get("violations", [])
            final["metadata"]["safety_events"] = self.safety.get_safety_stats()
            return final
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"Error processing query: {exc}", exc_info=True)
            return {
                "query": query,
                "error": str(exc),
                "response": f"An error occurred while processing your query: {exc}",
                "conversation_history": [],
                "metadata": {"error": True},
            }

    def _extract_results(self, query: str, state: ResearchState) -> Dict[str, Any]:
        conversation = state.get("conversation", [])
        draft = state.get("draft", "")
        critique = state.get("critique", "")

        final_response = draft or critique
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()

        num_sources = self._count_sources(conversation)
        agents_involved = list(
            {msg.get("source", "") for msg in conversation if msg.get("source")}
        )

        metadata = {
            "num_messages": len(conversation),
            "num_sources": num_sources,
            "agents_involved": agents_involved,
            "research_findings": state.get("evidence", ""),
        }

        return {
            "query": query,
            "response": final_response,
            "conversation_history": conversation,
            "citations": self._extract_citations(conversation),
            "metadata": metadata,
        }

    @staticmethod
    def _count_sources(messages: List[Dict[str, str]]) -> int:
        urls = set()
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                try:
                    content = " ".join(content) if isinstance(content, list) else str(content)
                except Exception:
                    content = str(content)
            for url in re.findall(r"https?://[^\s<>\"]+", content):
                urls.add(url)
        return max(len(urls), 1) if messages else 0

    @staticmethod
    def _extract_citations(messages: List[Dict[str, str]]) -> List[str]:
        citations: List[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                try:
                    content = " ".join(content) if isinstance(content, list) else str(content)
                except Exception:
                    content = str(content)
            for url in re.findall(r"https?://[^\s<>" + r"{}|\^`\[\]]+", content):
                if url not in citations:
                    citations.append(url)
        return citations[:20]
