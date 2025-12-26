import json
import wave
from vosk import Model, KaldiRecognizer

from rdoai.stt.base import Transcript

class LocalVoskStt:
    """
    Offline Vosk transcription for a WAV file (final segments).
    Works with mono 16-bit PCM WAV recommended.
    """
    def __init__(self, model_path: str):
        self.model = Model(model_path)

    def transcribe(self, wav_path: str, language: str | None = None) -> Transcript:
        # Vosk reads raw PCM; we feed from the WAV file
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()

            # Basic guardrails
            if sampwidth != 2:
                raise ValueError(f"Expected 16-bit PCM WAV (sampwidth=2). Got sampwidth={sampwidth}")
            if n_channels != 1:
                # You can downmix earlier; for now require mono to keep it simple/consistent.
                raise ValueError(f"Expected mono WAV (1 channel). Got channels={n_channels}")

            rec = KaldiRecognizer(self.model, framerate)
            rec.SetWords(True)

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

            final = json.loads(rec.FinalResult())
            text = (final.get("text") or "").strip()

        return Transcript(text=text, language=language or "en")
