"""
LLM-as-a-Judge (Guardrails-style, using OpenAI-compatible client)

Evaluates system responses against configurable criteria defined in config.yaml:
models.judge (provider/name/max_tokens) and evaluation.criteria (list of {name, weight, description}).

Usage:
    import asyncio
    from src.evaluation.judge import LLMJudge
    import yaml, os
    from dotenv import load_dotenv

    load_dotenv()
    cfg = yaml.safe_load(open("config.yaml"))
    judge = LLMJudge(cfg)
    result = asyncio.run(
        judge.evaluate("What is AI?", "AI is ...", sources=[], ground_truth="Artificial intelligence ...")
    )
    print(result["overall_score"], result["criterion_scores"])
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from openai import OpenAI


class LLMJudge:
    """LLM-based judge for evaluating system responses."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")
        self.model_config = config.get("models", {}).get("judge", {})
        self.criteria = config.get("evaluation", {}).get("criteria", [])
        eval_cfg = config.get("evaluation", {})
        judges_cfg = eval_cfg.get("judges", [])

        provider = self.model_config.get("provider", "openai")
        model_name = self.model_config.get("name", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else os.getenv("GROQ_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") if provider == "openai" else "https://api.groq.com/openai/v1"
        if not api_key:
            raise ValueError(f"Missing API key for judge provider '{provider}'")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_tokens = self.model_config.get("max_tokens", 1024)

        # Support dual judges (or more) with different perspectives/system prompts
        if judges_cfg and isinstance(judges_cfg, list):
            self.judge_variants = [
                {
                    "name": j.get("name", f"judge_{idx+1}"),
                    "system_prompt": j.get(
                        "system_prompt",
                        "You are a strict evaluator. Respond with a JSON object.",
                    ),
                }
                for idx, j in enumerate(judges_cfg)
            ]
        else:
            # Sensible defaults for two perspectives
            self.judge_variants = [
                {
                    "name": "factual_strict",
                    "system_prompt": "You are a strict evaluator focused on factual accuracy and evidence. Respond with JSON.",
                },
                {
                    "name": "user_value",
                    "system_prompt": "You are a pragmatic evaluator focused on usefulness, clarity, and safety. Respond with JSON.",
                },
            ]

        self.logger.info(
            "LLMJudge initialized with %d criteria, %d judge variants, model=%s",
            len(self.criteria),
            len(self.judge_variants),
            model_name,
        )

    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a response using LLM-as-a-Judge."""
        total_weight = sum(c.get("weight", 1.0) for c in self.criteria) or 1.0
        weighted = 0.0
        criterion_scores: Dict[str, Dict[str, Any]] = {}

        for criterion in self.criteria:
            name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)
            per_judge_results: List[Dict[str, Any]] = []

            for judge_variant in self.judge_variants:
                score_data = await self._judge_single(
                    criterion,
                    judge_variant=judge_variant,
                    query=query,
                    response=response,
                    sources=sources or [],
                    ground_truth=ground_truth or "",
                )
                per_judge_results.append(score_data)

            avg_score = (
                sum(r.get("score", 0.0) for r in per_judge_results) / len(per_judge_results)
                if per_judge_results
                else 0.0
            )
            criterion_scores[name] = {
                "score": avg_score,
                "judges": per_judge_results,
            }
            weighted += avg_score * weight

        overall = weighted / total_weight
        return {
            "query": query,
            "overall_score": overall,
            "criterion_scores": criterion_scores,
            "feedback": [v.get("reasoning", "") for v in criterion_scores.values()],
        }

    async def _judge_single(
        self,
        criterion: Dict[str, Any],
        judge_variant: Dict[str, Any],
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        ground_truth: str,
    ) -> Dict[str, Any]:
        name = criterion.get("name", "unknown")
        desc = criterion.get("description", "")
        prompt = self._build_prompt(name, desc, query, response, sources, ground_truth)
        try:
            judgment = self._call_llm(prompt, system_prompt=judge_variant.get("system_prompt"))
            score, reasoning = self._parse_judgment(judgment)
            return {
                "score": score,
                "reasoning": reasoning,
                "criterion": name,
                "judge": judge_variant.get("name", "judge"),
            }
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Judge failed for %s (%s): %s",
                name,
                judge_variant.get("name", "judge"),
                exc,
                exc_info=True,
            )
            return {
                "score": 0.0,
                "reasoning": f"error: {exc}",
                "criterion": name,
                "judge": judge_variant.get("name", "judge"),
            }

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                    or "You are a strict evaluator. Respond with a JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=self.max_tokens,
        )
        content = resp.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("Judge LLM returned empty content")
        return content

    @staticmethod
    def _parse_judgment(judgment: str) -> Tuple[float, str]:
        """
        Parse the judge output. Expect either a JSON-like "score: X, reasoning: Y"
        or a simple text; attempt to extract a numeric score in [0,1].
        """
        import json
        score = 0.0
        reasoning = judgment
        try:
            data = json.loads(judgment)
            score = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", reasoning)
        except Exception:
            # Fallback: naive pattern
            import re

            m = re.search(r"([01](?:\.\d+)?)", judgment)
            if m:
                score = float(m.group(1))
        score = max(0.0, min(1.0, score))
        return score, reasoning

    @staticmethod
    def _build_prompt(
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        ground_truth: str,
    ) -> str:
        src_note = f"{len(sources)} source(s) provided." if sources else "No sources provided."
        return f"""You are an expert evaluator. Score the response for the criterion below on a strict 0.0 to 1.0 scale.

Criterion: {criterion_name}
Description: {description}
Original Query: {query}
System Response:
\"\"\"{response}\"\"\"
Ground Truth (if any):
\"\"\"{ground_truth}\"\"\"
Sources note: {src_note}

Return JSON: {{"score": <0 to 1>, "reasoning": "<brief reason>"}}."""
