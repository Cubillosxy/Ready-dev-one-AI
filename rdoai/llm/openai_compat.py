from openai import OpenAI
from rdoai.llm.base import LlmResponse, ChatMessage

class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat(self, messages: list[ChatMessage], temperature: float = 0.3) -> LlmResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        return LlmResponse(text=text.strip())
