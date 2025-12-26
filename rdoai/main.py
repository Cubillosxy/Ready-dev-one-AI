from rdoai.config import AppConfig
from rdoai.app.controller import AppController

def main() -> None:
    cfg = AppConfig()
    AppController(cfg).run()
