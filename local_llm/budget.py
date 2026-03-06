"""Prompt and output budgeting helpers."""

from __future__ import annotations


class PromptBudgetError(RuntimeError):
    """Raised when a request exceeds the active profile budget."""


def validate_prompt_budget(
    *,
    prompt_tokens: int,
    max_output_tokens: int,
    default_context: int,
    hard_context: int,
    safe_mode: bool = True,
) -> dict:
    """Validate prompt/output size against the active profile limits."""
    requested_total = prompt_tokens + max_output_tokens
    allowed_context = default_context if safe_mode else hard_context

    if prompt_tokens >= hard_context:
        raise PromptBudgetError(
            f"Prompt is {prompt_tokens} tokens, which exceeds the hard limit of {hard_context}."
        )
    if requested_total > hard_context:
        raise PromptBudgetError(
            f"Prompt plus output ({requested_total}) exceeds the hard limit of {hard_context}."
        )
    if requested_total > allowed_context:
        raise PromptBudgetError(
            f"Prompt plus output ({requested_total}) exceeds the active context limit of {allowed_context}."
        )

    return {
        "prompt_tokens": prompt_tokens,
        "max_output_tokens": max_output_tokens,
        "allowed_context": allowed_context,
        "hard_context": hard_context,
        "requested_total": requested_total,
    }
