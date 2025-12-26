from rdoai.stt.base import Transcript

class LocalStubStt:
    def transcribe(self, wav_path: str, language: str | None = None) -> Transcript:
        # Stub for local models. Replace with your local pipeline.
        return Transcript(text=f"[LOCAL STT STUB] {wav_path}", language=language)


