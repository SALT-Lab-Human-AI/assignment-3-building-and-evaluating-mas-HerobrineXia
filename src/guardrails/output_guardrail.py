"""
Output Guardrail
Checks system outputs for safety violations.
"""

from typing import Dict, Any, List, Optional
import re

from nemoguardrails import RailsConfig, LLMRails  # type: ignore


class OutputGuardrail:
    """
    Guardrail for checking output safety.
    Provides PII detection, harmful content checks, and optional NeMo moderation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.prohibited_categories = config.get("prohibited_categories", [])

        self.nemo_guard = None
        framework = config.get("framework", "")
        if framework == "nemo_guardrails":
            categories_str = ", ".join(self.prohibited_categories)
            rails_spec = f"""
rails:
  config:
    tracing: false
  output:
    flows:
      - moderation_output
prompts:
  - task: moderation_output
    content: |
      You are a safety moderator. If the text contains any of these categories: {categories_str}, respond with "UNSAFE" and list categories. Otherwise respond "SAFE".
      Text: {{assistant}}
"""
            rails_config = RailsConfig.from_content(rails_spec)
            self.nemo_guard = LLMRails(config=rails_config)

    def validate(self, response: str, sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Validation result with valid flag, violations, and sanitized output
        """
        violations: List[Dict[str, Any]] = []

        # NeMo moderation
        nemo_violations = self._moderate_with_nemo(response)
        if nemo_violations is not None:
            violations.extend(nemo_violations)

        # PII and harmful content checks
        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))

        # Lightweight consistency placeholder (non-blocking)
        if sources:
            violations.extend(self._check_factual_consistency(response, sources))

        valid = len(violations) == 0
        sanitized = self._sanitize(response, violations) if violations else response

        return {
            "valid": valid,
            "violations": violations,
            "sanitized_output": sanitized
        }

    def _moderate_with_nemo(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Run NeMo Guardrails moderation if configured; None if unavailable/error."""
        if not self.nemo_guard:
            return None
        try:
            result = self.nemo_guard.generate(messages=[{"role": "assistant", "content": text}], flow="moderation_output")
            content = ""
            if isinstance(result, dict) and "output" in result:
                content = result.get("output", "")
            elif hasattr(result, "output"):
                content = getattr(result, "output")
            else:
                content = str(result)
            if isinstance(content, str) and content.strip().upper().startswith("UNSAFE"):
                return [{
                    "validator": "nemo_guardrails",
                    "reason": "NeMo Guardrails marked output as unsafe.",
                    "severity": "high"
                }]
            return []
        except Exception:  # pragma: no cover
            return None

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """Check for personally identifiable information."""
        violations: List[Dict[str, Any]] = []

        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        }

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Contains {pii_type}",
                    "severity": "high",
                    "matches": matches
                })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """Check for harmful or inappropriate content."""
        violations: List[Dict[str, Any]] = []

        harmful_keywords = [
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
        lowered = text.lower()
        for keyword in harmful_keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", lowered, flags=re.IGNORECASE):
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"May contain harmful content: {keyword}",
                    "severity": "medium"
                })

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Placeholder factual consistency check (non-blocking)."""
        # Could be extended with LLM verification against sources
        return []

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Sanitize text by removing/redacting violations."""
        sanitized = text

        for violation in violations:
            if violation.get("validator") == "pii":
                for match in violation.get("matches", []):
                    sanitized = sanitized.replace(match, "[REDACTED]")

        return sanitized
