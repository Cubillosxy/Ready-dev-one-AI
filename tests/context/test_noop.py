"""Tests for NoopContextProvider."""
from rdoai.context.noop import NoopContextProvider

def test_noop_context_provider():
    provider = NoopContextProvider()
    assert provider.build_context("any question") == ""
    assert provider.build_context("") == ""
