"""Tests for NoopContextProvider."""
from rdoai.context.noop import NoopContextProvider

def test_noop_context_provider():
    provider = NoopContextProvider()
    # NoopContextProvider returns a user profile context, not an empty string
    context = provider.build_context("any question")
    assert isinstance(context, str)
    assert len(context) > 0
    assert "Senior AI & Backend Software Engineer" in context
    
    # Should return same context regardless of input
    assert provider.build_context("") == context
