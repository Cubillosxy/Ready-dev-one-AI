import tkinter as tk
from dataclasses import dataclass
from typing import Callable

@dataclass
class OverlayState:
    listening: bool
    level_db: float
    vad_state: str
    last_pause_sec: float
    last_capture: str
    device_label: str
    hint: str
    transcript_partial: str
    transcript_final: str
    suggestion: str

class OverlayWindow:
    def __init__(self, geometry: str, alpha: float, always_on_top: bool):
        self.root = tk.Tk()
        self.root.title("Ready-Developer-One:AI")

        self.root.overrideredirect(True)
        self.root.attributes("-alpha", alpha)
        if always_on_top:
            self.root.attributes("-topmost", True)

        self.root.configure(bg="#111111")
        self.root.geometry(geometry)

        self._drag = {"x": 0, "y": 0}
        self.root.bind("<ButtonPress-1>", self._on_press)
        self.root.bind("<B1-Motion>", self._on_drag)
        self.root.bind("<Escape>", lambda e: self.root.quit())

        self.device_var = tk.StringVar(value="Input device: (unknown)")
        self.status_var = tk.StringVar(value="Listening: ON")
        self.state_var = tk.StringVar(value="State: silence")
        self.level_var = tk.StringVar(value="Level: -160.0 dB")
        self.pause_var = tk.StringVar(value="Last pause: 0.00s")
        self.lastcap_var = tk.StringVar(value="Last capture: none")
        self.hint_var = tk.StringVar(value="Buttons work always. ESC quits.")
        self.suggestion_var = tk.StringVar(value="Suggestion: ...")

        self.partial_var = tk.StringVar(value="Partial: ...")
        self.final_var = tk.StringVar(value="Final: ...")

        font_s = ("Helvetica", 11)
        font_c = ("Helvetica", 10)
        font_hint = ("Helvetica", 8)


        # buttons
        self._on_toggle_listening: Callable[[], None] = lambda: None
        self._on_finalize: Callable[[], None] = lambda: None
        self._on_toggle_language: Callable[[], None] = lambda: None
        self._on_toggle_input: Callable[[], None] = lambda: None

        btn_row = tk.Frame(self.root, bg="#111111")
        btn_row.pack(fill="x", padx=10, pady=(10, 6))

        tk.Button(btn_row, text="Toggle Listening", command=lambda: self._on_toggle_listening())\
            .pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="Answer Now", command=lambda: self._on_finalize())\
            .pack(side="left")

        tk.Button(btn_row, text="Lang: EN/ES", command=lambda: self._on_toggle_language()).pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="Input: Meet/Mic", command=lambda: self._on_toggle_input()).pack(side="left", padx=(0, 8))


        bar = tk.Frame(self.root, bg="#111111")
        bar.pack(fill="x", padx=10, pady=(0, 6))

        tk.Label(bar, textvariable=self.device_var, fg="white", bg="#111111", font=font_c)\
            .pack(side="left", padx=(0, 10))

        tk.Label(bar, textvariable=self.status_var, fg="white", bg="#111111", font=font_c)\
            .pack(side="left", padx=(0, 10))

        tk.Label(bar, textvariable=self.state_var, fg="white", bg="#111111", font=font_c)\
            .pack(side="left", padx=(0, 10))

        tk.Label(bar, textvariable=self.level_var, fg="white", bg="#111111", font=font_c)\
            .pack(side="left", padx=(0, 10))

        tk.Label(bar, textvariable=self.pause_var, fg="white", bg="#111111", font=font_c)\
            .pack(side="left", padx=(0, 10))

        tk.Label(bar, textvariable=self.lastcap_var, fg="white", bg="#111111", font=font_c)\
            .pack(side="left")


        # 3) Suggestion (current) – keep it visible, but not duplicated
        tk.Label(self.root, textvariable=self.suggestion_var, fg="white", bg="#111111",
                font=font_s, wraplength=720, justify="left")\
            .pack(anchor="w", padx=10, pady=(6, 6))

        sep = tk.Frame(self.root, bg="#333333", height=1)
        sep.pack(fill="x", padx=10, pady=(6, 10))

        # 4) Partial/Final (optional; can be compact)
        tk.Label(self.root, textvariable=self.partial_var, fg="white", bg="#111111",
                font=font_c, wraplength=720, justify="left")\
            .pack(anchor="w", padx=10, pady=(0, 2))

        tk.Label(self.root, textvariable=self.final_var, fg="white", bg="#111111",
                font=font_c, wraplength=720, justify="left")\
            .pack(anchor="w", padx=10, pady=(0, 8))


        # 6) Hint footer
        tk.Label(self.root, textvariable=self.hint_var, fg="white", bg="#111111",
                font=font_hint)\
            .pack(anchor="w", padx=10, pady=(0, 10))

    def set_handlers(self, on_toggle_listening: Callable[[], None], on_finalize: Callable[[], None], on_toggle_language: Callable[[], None], on_toggle_input: Callable[[], None]) -> None:
        self._on_toggle_listening = on_toggle_listening
        self._on_finalize = on_finalize
        self._on_toggle_language = on_toggle_language
        self._on_toggle_input = on_toggle_input

    def render(self, state: OverlayState) -> None:
        self.device_var.set(state.device_label.replace("Input device:", "Input:"))
        self.status_var.set(f"Listening: {'ON' if state.listening else 'OFF'}")
        self.state_var.set(f"State: {state.vad_state}")
        self.level_var.set(f"Level: {state.level_db:.1f} dB")
        self.pause_var.set(f"Last pause: {state.last_pause_sec:.2f}s")
        self.lastcap_var.set(f"Last capture: {state.last_capture}")
        self.suggestion_var.set(f"Suggestion: {state.suggestion}")
        
        self.partial_var.set(f"Partial: {state.transcript_partial}")
        self.final_var.set(f"Final: {state.transcript_final}")
        

        self.hint_var.set(state.hint)

    def every(self, ms: int, fn) -> None:
        self.root.after(ms, fn)

    def mainloop(self) -> None:
        self.root.mainloop()

    def _on_press(self, event):
        self._drag["x"] = event.x
        self._drag["y"] = event.y

    def _on_drag(self, event):
        x = self.root.winfo_x() + (event.x - self._drag["x"])
        y = self.root.winfo_y() + (event.y - self._drag["y"])
        self.root.geometry(f"+{x}+{y}")
