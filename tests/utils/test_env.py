"""Comprehensive tests for environment variable utilities."""
import os
import pytest
from rdoai.utils.env import getenv_int, getenv_str, load_env


class TestGetenvInt:
    """Test getenv_int function."""

    def test_returns_int_when_present(self, monkeypatch):
        """Test that getenv_int returns int value when env var exists."""
        monkeypatch.setenv("TEST_INT", "42")
        result = getenv_int("TEST_INT", 10)
        assert result == 42
        assert isinstance(result, int)

    def test_returns_default_when_missing(self, monkeypatch):
        """Test that getenv_int returns default when env var is missing."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = getenv_int("MISSING_VAR", 99)
        assert result == 99

    def test_returns_default_when_empty_string(self, monkeypatch):
        """Test that getenv_int returns default when env var is empty."""
        monkeypatch.setenv("EMPTY_VAR", "")
        result = getenv_int("EMPTY_VAR", 7)
        assert result == 7

    def test_returns_default_when_whitespace_only(self, monkeypatch):
        """Test that getenv_int returns default when env var is whitespace."""
        monkeypatch.setenv("WHITESPACE_VAR", "   ")
        result = getenv_int("WHITESPACE_VAR", 15)
        assert result == 15

    def test_strips_whitespace_from_value(self, monkeypatch):
        """Test that getenv_int strips whitespace from value."""
        monkeypatch.setenv("PADDED_INT", "  123  ")
        result = getenv_int("PADDED_INT", 0)
        assert result == 123

    def test_handles_negative_numbers(self, monkeypatch):
        """Test that getenv_int handles negative numbers."""
        monkeypatch.setenv("NEGATIVE_INT", "-50")
        result = getenv_int("NEGATIVE_INT", 0)
        assert result == -50

    def test_raises_on_invalid_int(self, monkeypatch):
        """Test that getenv_int raises ValueError on invalid int string."""
        monkeypatch.setenv("INVALID_INT", "not_a_number")
        with pytest.raises(ValueError):
            getenv_int("INVALID_INT", 0)

    def test_handles_zero(self, monkeypatch):
        """Test that getenv_int handles zero correctly."""
        monkeypatch.setenv("ZERO_VAR", "0")
        result = getenv_int("ZERO_VAR", 100)
        assert result == 0


class TestGetenvStr:
    """Test getenv_str function."""

    def test_returns_string_when_present(self, monkeypatch):
        """Test that getenv_str returns string value when env var exists."""
        monkeypatch.setenv("TEST_STRING", "hello")
        result = getenv_str("TEST_STRING", "default")
        assert result == "hello"
        assert isinstance(result, str)

    def test_returns_default_when_missing(self, monkeypatch):
        """Test that getenv_str returns default when env var is missing."""
        monkeypatch.delenv("MISSING_STRING", raising=False)
        result = getenv_str("MISSING_STRING", "fallback")
        assert result == "fallback"

    def test_returns_default_when_empty_string(self, monkeypatch):
        """Test that getenv_str returns default when env var is empty."""
        monkeypatch.setenv("EMPTY_STRING", "")
        result = getenv_str("EMPTY_STRING", "default")
        assert result == "default"

    def test_returns_default_when_whitespace_only(self, monkeypatch):
        """Test that getenv_str returns default when env var is whitespace."""
        monkeypatch.setenv("WHITESPACE_STRING", "   ")
        result = getenv_str("WHITESPACE_STRING", "default")
        assert result == "default"

    def test_strips_whitespace_from_value(self, monkeypatch):
        """Test that getenv_str strips whitespace from value."""
        monkeypatch.setenv("PADDED_STRING", "  hello world  ")
        result = getenv_str("PADDED_STRING", "default")
        assert result == "hello world"

    def test_preserves_internal_whitespace(self, monkeypatch):
        """Test that getenv_str preserves internal whitespace."""
        monkeypatch.setenv("MULTI_WORD", "hello   world")
        result = getenv_str("MULTI_WORD", "default")
        assert result == "hello   world"

    def test_handles_special_characters(self, monkeypatch):
        """Test that getenv_str handles special characters."""
        monkeypatch.setenv("SPECIAL_STRING", "hello@#$%^&*()")
        result = getenv_str("SPECIAL_STRING", "default")
        assert result == "hello@#$%^&*()"


class TestLoadEnv:
    """Test load_env function."""

    def test_load_env_does_not_raise(self):
        """Test that load_env executes without errors."""
        # This is a simple smoke test since load_env just calls dotenv.load_dotenv()
        # The actual .env file loading is tested by the dotenv library itself
        try:
            load_env()
        except Exception as e:
            pytest.fail(f"load_env() raised {type(e).__name__}: {e}")

    def test_load_env_is_callable(self):
        """Test that load_env is callable."""
        assert callable(load_env)
