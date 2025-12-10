# Project Summary

This project implements a multi‑agent research assistant using LangGraph with four agents (Planner, Researcher, Writer, Critic), heuristic guardrails, and an LLM‑as‑Judge evaluation pipeline. It replaces the earlier AutoGen setup and removes NeMo guardrails in favor of lightweight, dependency‑free safety checks.

## Architecture

- Orchestrator: `src/langgraph_orchestrator.py` drives the LangGraph workflow.
- Agents:
  - Planner: `src/agents/planner_agent.py`
  - Researcher: `src/agents/researcher_agent.py` (runs web/paper search tools)
  - Writer: `src/agents/writer_agent.py` (drafts and revises based on Critic feedback)
  - Critic: `src/agents/critic_agent.py` (concise approvals/revisions)
- Shared helpers: `src/agents/langgraph_agents.py` (LLM client, prompt loading, tool wrappers).
- Tools: `src/tools/web_search.py`, `src/tools/paper_search.py` (paper search optional/disabled by default), `src/tools/citation_tool.py`.
- Safety: `src/guardrails/safety_manager.py` provides heuristic guardrails; keyword lists include adult/explicit content and self‑harm/weapon terms. Legacy guardrail stubs remain but the orchestrator uses `SafetyManager`.
- UI: CLI (`python main.py --mode cli`) and Streamlit web UI (`python main.py --mode web`).

## Safety/Guardrails (Heuristic)

- Input/output checks block weapons/violence, self‑harm, and adult/explicit content via keyword filters (see `safety.prohibited_keywords` in `config.yaml` and defaults in `SafetyManager`).
- Violations are logged (UTF‑8) and can be refused or sanitized based on `safety.on_violation`.
- Safety stats are attached to responses; Streamlit shows events safely even if entries are strings.

## Evaluation (Phase 4)

- Dual judges: `src/evaluation/judge.py` now runs two independent judge prompts (configurable under `evaluation.judges`). Criterion scores are averaged across judges; overall score is weighted by criterion.
- Evaluator: `src/evaluation/evaluator.py` runs batch evaluation; command `python main.py --mode evaluate` loads `data/example_queries.json` (or configured path) and saves reports to `outputs/`.
- Judge model uses `models.judge` in `config.yaml`; temperature parameters were removed to avoid API errors with `gpt-5-mini`.

## Configuration Highlights

- `config.yaml` uses the OpenAI‑compatible `gpt-5-mini` by default; temperatures removed. Increase `agents.writer.max_tokens` in the config to allow longer drafts if needed.
- `safety.framework` is set to `heuristic`; adult/explicit keywords are listed under `safety.prohibited_keywords`.
- Paper search remains optional/disabled by default (`tools.paper_search.enabled: false`).
- UI verbose tracing is enabled (`ui.verbose: true`).

## Reproduction Steps

1) Set environment variables: `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if needed). For paper search, also set Semantic Scholar key if required by your tool implementation.
2) Install dependencies: `pip install -r requirements.txt`.
3) Run CLI: `python main.py --mode cli`.
4) Run Web UI: `python main.py --mode web`.
5) Run Evaluation: `python main.py --mode evaluate` (writes results to `outputs/`).

## Notes for Final Report

- Document the heuristic guardrail coverage (adult/explicit, weapons/violence, self‑harm) and refusal behavior.
- Report dual‑judge scoring and how overall scores are averaged across judges and criteria.
- Include evaluation dataset description (`data/example_queries.json`) and how to reproduce the saved reports.
- Mention that paper search is optional/disabled by default; web search uses Tavily/Brave per `config.yaml`.
- UI shows safety stats and agent traces; terminal logs remain the authoritative trace for debugging.
