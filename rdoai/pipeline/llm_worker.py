from dataclasses import dataclass
from typing import Optional

from rdoai.pipeline.assistant import AssistantPipeline
from rdoai.llm.errors import LlmErrorInfo

@dataclass
class LlmJob:
    question: str
    lang: str = "en"

@dataclass
class LlmEvent:
    kind: str               # "answer" | "error"
    answer: str = ""
    error: Optional[LlmErrorInfo] = None
    question: str = ""

class LlmWorker:
    def __init__(self, cfg, publish):
        self.cfg = cfg
        self.publish = publish
        self.assistant = AssistantPipeline(cfg)
        self._last_sent = ""

    def handle(self, job: LlmJob) -> None:
        q = (job.question or "").strip()
        if not q:
            return

        # basic de-dupe (prevents spamming on same segment)
        if q == self._last_sent:
            return
        self._last_sent = q

        result = self.assistant.answer(q, lang=job.lang)
        if result.error:
            self.publish(LlmEvent(kind="error", error=result.error, question=q))
        else:
            self.publish(LlmEvent(kind="answer", answer=result.answer or "", question=q))
