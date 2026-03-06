"""Tests for prompt budgeting safeguards."""

import pytest

from local_llm.budget import PromptBudgetError, validate_prompt_budget


def test_validate_prompt_budget_accepts_safe_request():
    validate_prompt_budget(
        prompt_tokens=1200,
        max_output_tokens=512,
        default_context=4096,
        hard_context=8192,
        safe_mode=True,
    )


def test_validate_prompt_budget_rejects_safe_mode_overflow():
    with pytest.raises(PromptBudgetError):
        validate_prompt_budget(
            prompt_tokens=3900,
            max_output_tokens=512,
            default_context=4096,
            hard_context=8192,
            safe_mode=True,
        )


def test_validate_prompt_budget_rejects_hard_limit_overflow():
    with pytest.raises(PromptBudgetError):
        validate_prompt_budget(
            prompt_tokens=8000,
            max_output_tokens=512,
            default_context=4096,
            hard_context=8192,
            safe_mode=False,
        )
