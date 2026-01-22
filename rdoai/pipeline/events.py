from dataclasses import dataclass
import numpy as np

@dataclass
class AudioChunk:
    audio: np.ndarray   # mono float32
    ts: float

@dataclass
class FlushStt:
    reason: str = "segment_end"

@dataclass
class SegmentReady:
    wav_path: str
    duration_sec: float
    ts: float

@dataclass
class SttPartial:
    text: str

@dataclass
class SttFinal:
    text: str

@dataclass
class LlmResult:
    answer: str

@dataclass
class ErrorEvent:
    source: str
    message: str
    status_code: int | None = None
