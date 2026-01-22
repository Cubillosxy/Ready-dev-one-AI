from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from openai import (
    APIStatusError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)

from rdoai.llm.errors import LlmErrorInfo

ChatMessage = Dict[str, str]

@dataclass
class LlmResponse:
    text: str

class OpenAIChatClient:
    def __init__(self, base_url: str, api_key: str, model: str, max_retries: int = 2):
        # openai-python retries some errors (including 429) by default; configurable via max_retries. :contentReference[oaicite:1]{index=1}
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.model = model

    def chat(self, messages: List[ChatMessage], temperature: float = 0.3) -> Tuple[Optional[LlmResponse], Optional[LlmErrorInfo]]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            return LlmResponse(text=text), None

        except RateLimitError as e:
            # 429 can mean: rate limit reached OR insufficient quota. :contentReference[oaicite:2]{index=2}
            status = getattr(e, "status_code", 429)
            retry_after = _try_retry_after_seconds(e)
            return None, LlmErrorInfo(
                status_code=status,
                code="rate_limit",
                message=str(e),
                retry_after_sec=retry_after,
            )

        except APIStatusError as e:
            # Non-2xx (4xx/5xx). SDK exposes status_code/response. :contentReference[oaicite:3]{index=3}
            status = getattr(e, "status_code", None)
            retry_after = _try_retry_after_seconds(e)

            code = "api_status_error"
            if status == 401:
                code = "auth_error"
            elif status == 403:
                code = "permission_denied"
            elif status == 404:
                code = "not_found"
            elif status == 429:
                code = "rate_limit"  # same family; treat like rate limiting
            elif status is not None and status >= 500:
                code = "server_error"

            #print full headers sent to OpenAI
            print("----- error details -----")
            print(self.client.api_key)
            print(self.client.base_url)

            return None, LlmErrorInfo(
                status_code=status,
                code=code,
                message=str(e),
                retry_after_sec=retry_after,
            )

        except (APIConnectionError, APITimeoutError) as e:
            return None, LlmErrorInfo(
                status_code=None,
                code="network_error",
                message=str(e),
                retry_after_sec=None,
            )

        except Exception as e:
            return None, LlmErrorInfo(
                status_code=None,
                code="unknown_error",
                message=str(e),
                retry_after_sec=None,
            )


def _try_retry_after_seconds(exc: Exception) -> Optional[float]:
    """
    Best-effort extraction of Retry-After header if available.
    Not all errors include it, but when present it is helpful for 429.
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if not headers:
        return None

    # headers can be a dict-like object
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except Exception:
        return None
