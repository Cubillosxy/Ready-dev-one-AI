"""Tests for prompting templates."""
from rdoai.prompting.templates import build_user_prompt

def test_build_user_prompt_basic():
    q = "How does GC work?"
    prompt = build_user_prompt(q)
    assert q in prompt
    assert "Context:" not in prompt
    assert "spoken answer" in prompt

def test_build_user_prompt_with_context():
    q = "Explain this code"
    ctx = "def foo(): pass"
    prompt = build_user_prompt(q, extra_context=ctx)
    assert q in prompt
    assert "Context:" in prompt
    assert ctx in prompt

def test_build_user_prompt_spanish():
    q = "Hola"
    prompt = build_user_prompt(q, lang="es")
    assert "español" in prompt

def test_build_user_prompt_stripping():
    q = "  What is Python?  "
    prompt = build_user_prompt(q)
    # Check that it doesn't have the extra spaces in the template-filled part
    # (Checking if it's contained without leading spaces)
    assert "What is Python?\n" in prompt
