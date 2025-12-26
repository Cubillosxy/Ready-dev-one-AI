from rdoai.pipeline.events import AudioChunk, SttPartial, SttFinal, ErrorEvent, FlushStt
from rdoai.stt.vosk_streaming import VoskStreamingStt

class StreamingSttWorker:
    def __init__(self, model_path: str, sample_rate: int, publish):
        self.engine = VoskStreamingStt(model_path=model_path, sample_rate=sample_rate)
        self.publish = publish  # function(event)

    def on_audio(self, item) -> None:
        try:
            if isinstance(item, AudioChunk):
                out = self.engine.accept_audio(item.audio)
                if not out:
                    return
                if out.is_final:
                    self.publish(SttFinal(out.text))
                else:
                    self.publish(SttPartial(out.text))
                return

            if isinstance(item, FlushStt):
                txt = self.engine.flush_final()
                if txt:
                    self.publish(SttFinal(txt))
                return

        except Exception as e:
            self.publish(ErrorEvent(source="streaming_stt", message=str(e)))