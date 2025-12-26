import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from rdoai.config import AudioConfig

@dataclass
class AudioFrame:
    audio: np.ndarray   # mono float32
    frames: int
    timestamp: float

class AudioCapture:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.stream: Optional[sd.InputStream] = None
        self.on_frame: Optional[Callable[[AudioFrame], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

    def start(self) -> None:
        try:
            dev = sd.query_devices(self.cfg.device_index)
            if dev["max_input_channels"] < 1:
                raise RuntimeError(f"Device #{self.cfg.device_index} has no input channels.")

            self.stream = sd.InputStream(
                device=self.cfg.device_index,
                samplerate=self.cfg.sample_rate,
                blocksize=self.cfg.block_size,
                channels=self.cfg.channels,
                dtype=self.cfg.dtype,
                callback=self._callback,
            )
            self.stream.start()
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            else:
                raise

    def stop(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None

    def _callback(self, indata, frames, time_info, status):
        try:
            # Use channel 0 as mono source (works for BlackHole)
            audio = indata[:, 0].astype(np.float32)
            frame = AudioFrame(audio=audio, frames=frames, timestamp=time.time())
            if self.on_frame:
                self.on_frame(frame)
        except Exception as e:
            if self.on_error:
                self.on_error(e)
