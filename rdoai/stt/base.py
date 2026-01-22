from dataclasses import dataclass
from typing import Protocol, Optional, List

@dataclass
class Transcript:
    text: str
    language: Optional[str] = None

class SttClient(Protocol):
    def transcribe(self, wav_path: str, language: Optional[str] = None) -> Transcript:
        ...
