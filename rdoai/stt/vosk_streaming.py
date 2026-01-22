import json
import numpy as np
from vosk import Model, KaldiRecognizer

from rdoai.stt.streaming_base import StreamingResult

class VoskStreamingStt:
    def __init__(self, model_path: str, sample_rate: int):
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        self.rec = KaldiRecognizer(self.model, sample_rate)
        self.rec.SetWords(True)

        self._last_partial = ""
        self._last_final = ""

    def accept_audio(self, pcm_float32: np.ndarray) -> StreamingResult | None:
        # Vosk expects int16 bytes
        pcm_int16 = np.clip(pcm_float32, -1.0, 1.0)
        pcm_int16 = (pcm_int16 * 32767.0).astype(np.int16)
        data = pcm_int16.tobytes()

        is_final = self.rec.AcceptWaveform(data)

        if is_final:
            result = json.loads(self.rec.Result())
            text = (result.get("text") or "").strip()
            if text and text != self._last_final:
                self._last_final = text
                return StreamingResult(text=text, is_final=True)
            return None

        partial = json.loads(self.rec.PartialResult()).get("partial", "").strip()
        # throttle noise: only emit if it changes meaningfully
        if partial and partial != self._last_partial and len(partial) >= 3:
            self._last_partial = partial
            return StreamingResult(text=partial, is_final=False)

        return None

    def flush_final(self) -> str:
        """
        Force a final result from the recognizer (useful when our VAD says the utterance ended).
        Then reset recognizer state so next utterance starts clean.
        """
        try:
            result = json.loads(self.rec.FinalResult())
            text = (result.get("text") or "").strip()
        except Exception:
            text = ""

        # Reset recognizer by recreating it (most robust across versions)
        self.rec = KaldiRecognizer(self.model, self.sample_rate)
        self.rec.SetWords(True)
        self._last_partial = ""
        self._last_final = ""
        return text