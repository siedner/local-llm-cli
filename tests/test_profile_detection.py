"""Tests for profile detection parsing from sysctl outputs."""

from local_llm.config import match_profile


def test_m1_pro_32gb():
    assert match_profile("Apple M1 Pro", 32) == "m1pro32"


def test_m4_16gb():
    assert match_profile("Apple M4", 16) == "m4mini16"


def test_m1_pro_with_extra_text():
    """sysctl may return slightly different strings."""
    assert match_profile("Apple M1 Pro (10-core)", 32) == "m1pro32"


def test_m4_with_extra_text():
    assert match_profile("Apple M4 chip", 16) == "m4mini16"


def test_m1_pro_64gb():
    """Higher memory should still match."""
    assert match_profile("Apple M1 Pro", 64) == "m1pro32"


def test_m4_32gb():
    """M4 with more memory still matches M4 profile first."""
    result = match_profile("Apple M4", 32)
    assert result == "m432"


def test_unknown_chip_high_memory():
    """Unknown chip with enough memory falls back to memory-based match."""
    result = match_profile("Apple M3 Max", 96)
    assert result is not None


def test_very_low_memory():
    """4GB won't match any profile."""
    result = match_profile("Apple M1", 4)
    assert result is None
