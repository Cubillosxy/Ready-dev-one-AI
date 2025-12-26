from typing import Protocol

class ContextProvider(Protocol):
    def build_context(self, question_text: str) -> str:
        ...
