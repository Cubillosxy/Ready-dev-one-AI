import os

def repo_root() -> str:
    # paths.py is at rdoai/utils/paths.py. 
    # To get to root, we go up three levels: utils -> rdoai -> root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
