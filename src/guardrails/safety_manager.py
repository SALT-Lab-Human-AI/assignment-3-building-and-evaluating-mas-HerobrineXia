"""
Safety Manager (Guardrails AI style, heuristic-only)

This implementation removes NeMo dependencies and uses lightweight, rule-based
checks to block unsafe input/output. It logs safety events and applies the
configured on_violation policy.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import re


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.
    Heuristic-only: no external moderation models required.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.debug = config.get("debug", False)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories/keywords
        self.prohibited_categories = config.get(
            "prohibited_categories",
            ["harmful_content", "personal_attacks", "misinformation", "off_topic_queries"],
        )
        default_input = [
            "hack",
            "exploit",
            "bypass",
            "weapon",
            "gun",
            "firearm",
            "ammunition",
            "shoot",
            "grenade",
            "explosive",
            "bomb",
            "knife",
            "suicide",
            "self-harm",
            "porn",
            "pornography",
            "adult content",
            "nsfw",
            "sexual content",
            "explicit sexual content",
            "child sexual abuse material",
            "csam",
        ]
        default_output = [
            "violent",
            "harmful",
            "dangerous",
            "weapon",
            "gun",
            "firearm",
            "ammunition",
            "shoot",
            "grenade",
            "explosive",
            "bomb",
            "knife",
            "suicide",
            "self-harm",
            "porn",
            "pornography",
            "adult content",
            "nsfw",
            "sexual content",
            "explicit sexual content",
            "child sexual abuse material",
            "csam",
        ]
        # Allow config to override or extend
        cfg_inputs = config.get("prohibited_keywords", [])
        cfg_outputs = config.get("harmful_output_keywords", [])
        self.prohibited_keywords = list({k.lower(): k for k in (default_input + cfg_inputs)}.keys())
        self.harmful_output_keywords = list({k.lower(): k for k in (default_output + cfg_outputs)}.keys())

        # Violation response strategy
        self.on_violation = config.get("on_violation", {})

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """Check if input query is safe to process."""
        if not self.enabled:
            return {"safe": True}

        violations = self._heuristic_input_checks(query)
        is_safe = len(violations) == 0
        if self.debug:
            self.logger.info("Input safety check: safe=%s violations=%s", is_safe, violations)

        if not is_safe and self.log_events:
            self._log_safety_event("input", query, violations, is_safe)

        return {"safe": is_safe, "violations": violations}

    def check_output_safety(
        self, response: str, sources: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Check if output response is safe to return."""
        if not self.enabled:
            return {"safe": True, "response": response}

        violations = self._heuristic_output_checks(response, sources)
        is_safe = len(violations) == 0
        if self.debug:
            self.logger.info("Output safety check: safe=%s violations=%s", is_safe, violations)

        if not is_safe and self.log_events:
            self._log_safety_event("output", response, violations, is_safe)

        result = {"safe": is_safe, "violations": violations, "response": response}

        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            if action == "sanitize":
                result["response"] = self._sanitize_response(response, violations)
            elif action == "refuse":
                result["response"] = self.on_violation.get(
                    "message", "I cannot provide this response due to safety policies."
                )

        return result

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """Sanitize response by redacting unsafe segments."""
        sanitized = response
        for violation in violations:
            reason = violation.get("reason", "")
            if reason:
                sanitized = sanitized.replace(reason, "[REDACTED]")
        return sanitized

    def _log_safety_event(
        self, event_type: str, content: str, violations: List[Dict[str, Any]], is_safe: bool
    ):
        """Log a safety event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
        }
        self.safety_events.append(event)
        self.logger.warning("Safety event: type=%s safe=%s violations=%s", event_type, is_safe, violations)

        log_file = self.config.get("safety_log_file")
        if log_file and self.log_events:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:  # pragma: no cover
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])
        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0,
        }

    def clear_events(self):
        self.safety_events = []

    def _heuristic_input_checks(self, query: str) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        for keyword in self.prohibited_keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", query, flags=re.IGNORECASE):
                if self.debug:
                    self.logger.info("Input keyword hit: %s", keyword)
                violations.append(
                    {
                        "category": "potentially_harmful",
                        "reason": f"Query contains prohibited keyword: {keyword}",
                        "severity": "medium",
                    }
                )
        return violations

    def _heuristic_output_checks(
        self, response: str, sources: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        for keyword in self.harmful_output_keywords:
            # Use word-boundary style match to reduce false positives (e.g., "begun" vs "gun")
            if re.search(rf"\b{re.escape(keyword)}\b", response, flags=re.IGNORECASE):
                if self.debug:
                    self.logger.info("Output keyword hit: %s", keyword)
                violations.append(
                    {
                        "validator": "harmful_content",
                        "reason": f"May contain harmful content: {keyword}",
                        "severity": "medium",
                    }
                )
        return violations
