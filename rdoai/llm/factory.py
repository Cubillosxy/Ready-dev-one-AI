from rdoai.config import LlmConfig
from rdoai.llm.base import LlmClient
#from rdoai.llm.openai_compat import OpenAICompatClient
from rdoai.llm.openai_client import OpenAIChatClient


def build_llm(cfg: AppConfig) -> OpenAIChatClient:
    # Provider-agnostic entry point: later you can add lmstudio/local clients here.
    return OpenAIChatClient(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        max_retries=2,
    )