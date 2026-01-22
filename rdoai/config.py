from dataclasses import dataclass
from rdoai.utils.env import load_env, getenv_int, getenv_str

import os

# UI config
WIDTH = 820
HEIGHT = 680


load_env()


def getenv_float(key: str, default: float) -> float:
    v = os.getenv(key)
    return float(v) if v and v.strip() else default

@dataclass(frozen=True)
class InputConfig:
    mode: str = getenv_str("RDOAI_INPUT_MODE", "meeting")  # meeting | mic
    meeting_device_index: int = int(getenv_str("RDOAI_MEETING_DEVICE_INDEX", "1"))
    mic_device_index: int = int(getenv_str("RDOAI_MIC_DEVICE_INDEX", "2"))

@dataclass(frozen=True)
class LanguageConfig:
    lang: str = getenv_str("RDOAI_LANG", "en")  # en | es
    vosk_model_en: str = getenv_str("RDOAI_VOSK_MODEL_EN", "models/vosk-model-small-en-us-0.15")
    vosk_model_es: str = getenv_str("RDOAI_VOSK_MODEL_ES", "models/vosk-model-small-es-0.42")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    block_size: int = 1024
    device_index: int = getenv_int("RDOAI_AUDIO_DEVICE_INDEX", 1)
    channels: int = 1
    dtype: str = "float32"

@dataclass(frozen=True)
class VadConfig:
    silence_threshold_db: float = -45.0
    speech_threshold_db: float = -40.0
    min_silence_sec: float = 0.6

@dataclass(frozen=True)
class MicVadConfig:
    silence_threshold_db: float = -50.0
    speech_threshold_db: float = -45.0
    min_silence_sec: float = 2.0

@dataclass(frozen=True)
class UiConfig:
    window_alpha: float = 0.88
    always_on_top: bool = True
    geometry: str = f"{WIDTH}x{HEIGHT}+60+60"   

@dataclass(frozen=True)
class SttConfig:
    provider: str = getenv_str("RDOAI_STT_PROVIDER", "openai")  # openai | local
    openai_model: str = getenv_str("RDOAI_OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    language: str = getenv_str("RDOAI_OPENAI_STT_LANGUAGE", "en")
    vosk_model_path: str = getenv_str("RDOAI_VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")

# Streaming STT config

@dataclass(frozen=True)
class StreamingSttConfig:
    provider: str = getenv_str("RDOAI_STREAMING_STT_PROVIDER", "vosk")  # vosk | off
    vosk_model_path: str = getenv_str("RDOAI_VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")

@dataclass(frozen=True)
class LlmConfig:
    provider: str = getenv_str("RDOAI_LLM_PROVIDER", "openai")  # lmstudio | openai_compat
    base_url: str = getenv_str("RDOAI_LLM_BASE_URL", "http://localhost:1234/v1")
    api_key: str = getenv_str("OPENAI_API_KEY", "")
    model: str = getenv_str("RDOAI_LLM_MODEL", "openai/gpt-oss-20b")
    temperature: float = getenv_float("RDOAI_LLM_TEMPERATURE", 0.3)

@dataclass(frozen=True)
class AssistantConfig:
    trigger: str = getenv_str("RDOAI_ASSIST_TRIGGER", "on_pause")  # on_pause | manual



@dataclass(frozen=True)
class AppConfig:
    output_dir: str = "captures"
    debug_print_every_sec: float = 0.5
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    mic_vad: MicVadConfig = MicVadConfig()
    ui: UiConfig = UiConfig()
    stt: SttConfig = SttConfig()
    streaming_stt: StreamingSttConfig = StreamingSttConfig()
    llm: LlmConfig = LlmConfig()
    assistant: AssistantConfig = AssistantConfig()
    input: InputConfig = InputConfig()
    lang: LanguageConfig = LanguageConfig()

