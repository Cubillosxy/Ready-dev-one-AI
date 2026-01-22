import threading
import queue
from typing import Callable, Any, Optional

class Worker:
    def __init__(self, name: str, target: Callable[[Any], None], maxsize: int = 200):
        self.name = name
        self.q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._target = target
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        # Unblock if waiting
        try:
            self.q.put_nowait(None)
        except Exception:
            pass

    def submit(self, item: Any) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            # drop if saturated to keep UI realtime
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self.q.get()
            if item is None:
                continue
            try:
                self._target(item)
            except Exception:
                # swallow to avoid killing thread; errors should be reported via results queue
                pass
