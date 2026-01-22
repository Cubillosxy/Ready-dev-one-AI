from dataclasses import dataclass
from typing import Optional

@dataclass
class LlmErrorInfo:
    status_code: Optional[int]
    code: str                 # e.g. rate_limit, insufficient_quota, auth_error, server_error, network_error
    message: str
    retry_after_sec: Optional[float] = None
