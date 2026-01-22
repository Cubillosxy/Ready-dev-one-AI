"""Tests for LlmErrorInfo dataclass."""
from rdoai.llm.errors import LlmErrorInfo


class TestLlmErrors:
    def test_llm_error_info_creation(self):
        info = LlmErrorInfo(
            status_code=429,
            code="rate_limit",
            message="Too many requests",
            retry_after_sec=10.5
        )
        
        assert info.status_code == 429
        assert info.code == "rate_limit"
        assert info.message == "Too many requests"
        assert info.retry_after_sec == 10.5

    def test_llm_error_info_defaults(self):
        info = LlmErrorInfo(
            status_code=None,
            code="network_error",
            message="Connection failed"
        )
        assert info.retry_after_sec is None
