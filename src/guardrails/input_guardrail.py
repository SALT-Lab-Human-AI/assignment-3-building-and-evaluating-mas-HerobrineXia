"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List


class InputGuardrail:
    """
    Guardrail for checking input safety.

    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result
        """
        violations: List[Dict[str, Any]] = []

        # Length checks
        length_issues = self._check_length(query)
        violations.extend(length_issues)

        # Prompt injection checks
        injection_issues = self._check_prompt_injection(query)
        violations.extend(injection_issues)

        # Toxic / harmful language checks
        toxic_issues = self._check_toxic_language(query)
        violations.extend(toxic_issues)

        # Relevance check (optional, uses configured topic if provided)
        relevance_issues = self._check_relevance(query)
        violations.extend(relevance_issues)

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_input": query  # Could be modified for stricter sanitization
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language.

        """
        violations: List[Dict[str, Any]] = []
        # Simple keyword-based filter; for production replace with a classifier
        toxic_keywords = [
            "kill", "hate", "terrorist", "attack", "bomb",
            "racist", "sexist", "suicide", "self-harm",
            "sex", "porn", "pornography", "nsfw", "explicit", "adult content", "child sexual",
        ]
        lowered = text.lower()
        for keyword in toxic_keywords:
            if keyword in lowered:
                violations.append({
                    "validator": "toxic_language",
                    "reason": f"Potentially harmful intent detected: '{keyword}'",
                    "severity": "high"
                })
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.

        """
        violations: List[Dict[str, Any]] = []
        # Check for common prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard",
            "forget everything",
            "system:",
            "sudo",
            "assistant:",
            "you are now",
            "override",
            "forget prior rules"
        ]

        for pattern in injection_patterns:
            if pattern.lower() in text.lower():
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection: {pattern}",
                    "severity": "high"
                })

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to the system's purpose.

        """
        violations: List[Dict[str, Any]] = []
        topic = self.config.get("system", {}).get("topic")
        if topic:
            # Basic topicality check: ensure some overlap with configured topic keywords
            if topic.lower() not in query.lower():
                violations.append({
                    "validator": "relevance",
                    "reason": f"Query may be off-topic (expected topic includes '{topic}')",
                    "severity": "low"
                })
        return violations

    def _check_length(self, text: str) -> List[Dict[str, Any]]:
        """Check for overly short or long inputs."""
        violations: List[Dict[str, Any]] = []
        if len(text.strip()) < 5:
            violations.append({
                "validator": "length",
                "reason": "Query too short",
                "severity": "low"
            })
        if len(text) > 2000:
            violations.append({
                "validator": "length",
                "reason": "Query too long",
                "severity": "medium"
            })
        return violations
