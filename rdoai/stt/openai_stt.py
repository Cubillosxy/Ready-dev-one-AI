from openai import OpenAI
from rdoai.stt.base import Transcript

class OpenAIStt:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    def transcribe(self, wav_path: str, language: str | None = None) -> Transcript:
        # The OpenAI speech-to-text guide describes the transcriptions API; models include
        # gpt-4o-transcribe and gpt-4o-mini-transcribe. :contentReference[oaicite:1]{index=1}
        with open(wav_path, "rb") as f:
            # Note: parameter names can differ slightly by SDK version; this follows the current guide pattern.
            resp = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                language=language,
            )

        # resp.text is the full transcript text (see transcription object). :contentReference[oaicite:2]{index=2}
        return Transcript(text=getattr(resp, "text", str(resp)), language=language)
