from rdoai.config import SttConfig
from rdoai.stt.base import SttClient
from rdoai.stt.openai_stt import OpenAIStt
from rdoai.stt.local_stub import LocalStubStt
from rdoai.stt.local_vosk import LocalVoskStt

def build_stt(cfg: SttConfig) -> SttClient:
    provider = (cfg.provider or "").strip().lower()
    if provider == "openai":
        return OpenAIStt(model=cfg.openai_model)
    if provider == "local":
        #return LocalStubStt()
        return LocalVoskStt(model_path=cfg.vosk_model_path)
    raise ValueError(f"Unknown STT provider: {cfg.provider}")
