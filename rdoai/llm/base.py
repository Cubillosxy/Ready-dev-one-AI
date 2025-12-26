from dataclasses import dataclass
from typing import Protocol, List, Dict, Any

ChatMessage = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}

@dataclass
class LlmResponse:
    text: str

class LlmClient(Protocol):
    def chat(self, messages: List[ChatMessage], temperature: float = 0.3) -> LlmResponse:
        ...
