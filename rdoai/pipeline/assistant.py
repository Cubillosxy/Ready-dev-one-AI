from rdoai.llm.base import ChatMessage
from rdoai.llm.factory import build_llm
from rdoai.context.noop import NoopContextProvider
from rdoai.prompting.templates import SYSTEM_INTERVIEW_COACH, build_user_prompt
from rdoai.config import AppConfig


#import dataclass
from dataclasses import dataclass
from typing import Optional

@dataclass
class AssistantResult:
    answer: Optional[str]
    error: Optional[str]


class AssistantPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.llm = build_llm(cfg)
        self.ctx_provider = NoopContextProvider()

    def answer(self, question_text: str, lang: str = "en") -> AssistantResult:
        extra_context = self.ctx_provider.build_context(question_text)
        user_prompt = build_user_prompt(question_text, extra_context=extra_context, lang=lang)

        messages = [
            {"role": "system", "content": SYSTEM_INTERVIEW_COACH},
            {"role": "user", "content": user_prompt},
        ]

        resp, err = self.llm.chat(messages, temperature=self.cfg.llm.temperature)
        if err:
            return AssistantResult(answer=None, error=err)

        return AssistantResult(answer=resp.text if resp else "", error=None)

        