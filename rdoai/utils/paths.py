import os

def repo_root() -> str:
    # run.py is at repo root; rdoai/ is a sibling
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
