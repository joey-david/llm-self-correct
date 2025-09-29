from __future__ import annotations

from typing import Dict


class PromptBuilder:
    """Construct plain prompts from standardized calibration records."""

    def build(self, record: Dict) -> str:
        answer_type = (record.get("answer_type") or "open").lower()
        question = (record.get("question") or "").strip()
        options = record.get("options") or []

        if answer_type == "choice":
            parts = [f"Q: {question}"]
            if options:
                parts.append("Options:")
                for idx, opt in enumerate(options):
                    label = chr(ord("A") + idx) if idx < 26 else f"Option {idx+1}"
                    parts.append(f"{label}) {opt}")
            parts.append("Answer with the option text only.")
            parts.append("A:")
            return "\n".join(parts)

        if answer_type == "boolean":
            return f"Q: {question}\nAnswer yes or no.\nA:"

        # numeric and open default
        return f"Q: {question}\nAnswer briefly.\nA:"
