from rdoai.pipeline.events import SegmentReady, LlmResult, ErrorEvent
from rdoai.pipeline.assistant import AssistantPipeline  # tu pipeline con OpenAI error mapping
from rdoai.stt.factory import build_stt

class PipelineWorker:
    def __init__(self, cfg, publish):
        self.cfg = cfg
        self.publish = publish
        self.stt = build_stt(cfg.stt)
        self.assistant = AssistantPipeline(cfg)

        self._last_sent = ""

    def on_segment(self, seg: SegmentReady) -> None:
        try:
            tr = self.stt.transcribe(seg.wav_path, language=self.cfg.stt.language)
            q = (tr.text or "").strip()
            if not q:
                self.publish(ErrorEvent(source="pipeline", message="Empty transcript; skip."))
                return

            # Simple de-dupe (can be improved)
            if q == self._last_sent:
                return

            self._last_sent = q

            res = self.assistant.answer(q)
            if res.error:
                e = res.error
                self.publish(ErrorEvent(
                    source="llm",
                    message=f"{e.code}: {e.message}",
                    status_code=e.status_code
                ))
                return

            self.publish(LlmResult(answer=res.answer or ""))

        except Exception as e:
            self.publish(ErrorEvent(source="pipeline", message=str(e)))
