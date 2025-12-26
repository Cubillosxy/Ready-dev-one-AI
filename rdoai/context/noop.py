from rdoai.context.base import ContextProvider

class NoopContextProvider(ContextProvider):
    def build_context(self, question_text: str) -> str:
        return ""
