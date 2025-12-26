import os
from dotenv import load_dotenv

def load_env() -> None:
    # Loads .env if present
    load_dotenv()

def getenv_int(key: str, default: int) -> int:
    v = os.getenv(key)
    return int(v) if v and v.strip() else default

def getenv_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v.strip() if v and v.strip() else default
