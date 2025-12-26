from dataclasses import dataclass
from typing import Protocol

@dataclass
class StreamingResult:
    text: str
    is_final: bool  # partial vs final

class StreamingSttClient(Protocol):
    def accept_audio(self, pcm_float32) -> StreamingResult | None:
        """Feed mono float32 audio chunk, returns partial/final text (or None)."""
        ...
