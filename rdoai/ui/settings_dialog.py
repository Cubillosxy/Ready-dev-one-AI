import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class VadSettings:
    """VAD settings for a specific mode (Meeting or Mic)."""
    silence_threshold_db: float
    speech_threshold_db: float
    min_silence_sec: float


@dataclass
class UiSettings:
    """UI settings."""
    window_opacity: float


@dataclass
class AllSettings:
    """Complete settings for VAD and UI."""
    meeting: VadSettings
    mic: VadSettings
    ui: UiSettings


class SettingsDialog:
    """Dialog window for configuring VAD audio parameters with sliders."""
    
    def __init__(self, parent: tk.Tk, current_settings: AllSettings, on_apply: Callable[[AllSettings], None], on_close: Callable[[], None]):
        self.parent = parent
        self.on_apply = on_apply
        self.on_close = on_close
        self.result: Optional[AllSettings] = None
        self.applied = False  # Track if settings were applied
        self.last_opacity = current_settings.ui.window_opacity
        
        # Configure ttk styles for buttons
        self.style = ttk.Style()
        self.style.theme_use('default')
        
        # Custom style for Apply button (green)
        self.style.configure(
            'Apply.TButton',
            background='#4CAF50',
            foreground='white',
            borderwidth=0,
            focuscolor='none',
            font=('Helvetica', 10, 'bold'),
            padding=(15, 6)
        )
        self.style.map(
            'Apply.TButton',
            background=[('active', '#45a049')],
            foreground=[('active', 'white')]
        )
        
        # Custom style for Cancel button (red)
        self.style.configure(
            'Cancel.TButton',
            background='#f44336',
            foreground='white',
            borderwidth=0,
            focuscolor='none',
            font=('Helvetica', 10),
            padding=(15, 6)
        )
        self.style.map(
            'Cancel.TButton',
            background=[('active', '#da190b')],
            foreground=[('active', 'white')]
        )
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("600x600")  # Larger size
        self.dialog.configure(bg="#1e1e1e")
        self.dialog.resizable(False, False)
        
        # Make dialog stay on top of parent
        self.dialog.attributes("-topmost", True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog on parent
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (600 // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (600 // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        # Title
        title = tk.Label(
            self.dialog,
            text="Settings",
            font=("Helvetica", 12, "bold"),
            fg="white",
            bg="#1e1e1e"
        )
        title.pack(pady=(8, 5))
        
        # Create frames for Meeting, Mic, and UI settings
        self._create_meeting_section(current_settings.meeting)
        self._create_separator()
        self._create_mic_section(current_settings.mic)
        self._create_separator()
        self._create_ui_section(current_settings.ui)
        
        # Buttons
        self._create_buttons()
        
        # Handle window close button and focus events
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Auto-apply when focus returns to main window
        self.dialog.bind("<FocusOut>", self._on_focus_out)
    
    def _create_meeting_section(self, settings: VadSettings):
        """Create Meeting mode settings section."""
        frame = tk.Frame(self.dialog, bg="#1e1e1e")
        frame.pack(fill="x", padx=20, pady=(5, 3))
        
        label = tk.Label(
            frame,
            text="Meeting Mode",
            font=("Helvetica", 10, "bold"),
            fg="#4CAF50",
            bg="#1e1e1e"
        )
        label.pack(anchor="w", pady=(0, 5))
        
        self.meeting_silence_var = tk.DoubleVar(value=settings.silence_threshold_db)
        self.meeting_speech_var = tk.DoubleVar(value=settings.speech_threshold_db)
        self.meeting_min_silence_var = tk.DoubleVar(value=settings.min_silence_sec)
        
        self._create_slider(
            frame,
            "Silence Threshold (dB):",
            self.meeting_silence_var,
            from_=-60.0,
            to=-20.0,
            resolution=1.0
        )
        
        self._create_slider(
            frame,
            "Speech Threshold (dB):",
            self.meeting_speech_var,
            from_=-60.0,
            to=-20.0,
            resolution=1.0
        )
        
        self._create_slider(
            frame,
            "Min Silence Duration (s):",
            self.meeting_min_silence_var,
            from_=0.1,
            to=3.0,
            resolution=0.1
        )
    
    def _create_mic_section(self, settings: VadSettings):
        """Create Mic mode settings section."""
        frame = tk.Frame(self.dialog, bg="#1e1e1e")
        frame.pack(fill="x", padx=20, pady=(5, 3))
        
        label = tk.Label(
            frame,
            text="Mic Mode",
            font=("Helvetica", 10, "bold"),
            fg="#2196F3",
            bg="#1e1e1e"
        )
        label.pack(anchor="w", pady=(0, 5))
        
        self.mic_silence_var = tk.DoubleVar(value=settings.silence_threshold_db)
        self.mic_speech_var = tk.DoubleVar(value=settings.speech_threshold_db)
        self.mic_min_silence_var = tk.DoubleVar(value=settings.min_silence_sec)
        
        self._create_slider(
            frame,
            "Silence Threshold (dB):",
            self.mic_silence_var,
            from_=-60.0,
            to=-20.0,
            resolution=1.0
        )
        
        self._create_slider(
            frame,
            "Speech Threshold (dB):",
            self.mic_speech_var,
            from_=-60.0,
            to=-20.0,
            resolution=1.0
        )
        
        self._create_slider(
            frame,
            "Min Silence Duration (s):",
            self.mic_min_silence_var,
            from_=0.1,
            to=5.0,
            resolution=0.1
        )
    
    def _create_ui_section(self, settings: UiSettings):
        """Create UI settings section."""
        frame = tk.Frame(self.dialog, bg="#1e1e1e")
        frame.pack(fill="x", padx=20, pady=(5, 3))
        
        label = tk.Label(
            frame,
            text="Window Settings",
            font=("Helvetica", 10, "bold"),
            fg="#FF9800",
            bg="#1e1e1e"
        )
        label.pack(anchor="w", pady=(0, 5))
        
        self.ui_opacity_var = tk.DoubleVar(value=settings.window_opacity)
        
        # Trace opacity changes to apply in real-time and keep dialog on top
        self.ui_opacity_var.trace_add("write", self._on_opacity_change)
        
        self._create_slider(
            frame,
            "Window Opacity:",
            self.ui_opacity_var,
            from_=0.5,
            to=1.0,
            resolution=0.05
        )
    
    def _create_slider(
        self,
        parent: tk.Frame,
        label_text: str,
        variable: tk.DoubleVar,
        from_: float,
        to: float,
        resolution: float
    ):
        """Create a labeled slider with value display."""
        container = tk.Frame(parent, bg="#1e1e1e")
        container.pack(fill="x", pady=3)
        
        # Label and value display
        label_frame = tk.Frame(container, bg="#1e1e1e")
        label_frame.pack(fill="x")
        
        label = tk.Label(
            label_frame,
            text=label_text,
            font=("Helvetica", 9),
            fg="white",
            bg="#1e1e1e"
        )
        label.pack(side="left")
        
        value_label = tk.Label(
            label_frame,
            textvariable=variable,
            font=("Helvetica", 9, "bold"),
            fg="#FFC107",
            bg="#1e1e1e",
            width=8,
            anchor="e"
        )
        value_label.pack(side="right")
        
        # Slider
        slider = tk.Scale(
            container,
            variable=variable,
            from_=from_,
            to=to,
            resolution=resolution,
            orient="horizontal",
            showvalue=False,
            bg="#2e2e2e",
            fg="white",
            highlightthickness=0,
            troughcolor="#424242",
            activebackground="#4CAF50"
        )
        slider.pack(fill="x", pady=(2, 0))
    
    def _create_separator(self):
        """Create a visual separator between sections."""
        sep = tk.Frame(self.dialog, bg="#424242", height=1)
        sep.pack(fill="x", padx=20, pady=8)
    
    def _create_buttons(self):
        """Create Apply and Cancel buttons."""
        button_frame = tk.Frame(self.dialog, bg="#1e1e1e")
        button_frame.pack(fill="x", padx=20, pady=(10, 15))
        
        cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            style='Cancel.TButton',
            cursor="hand2"
        )
        cancel_btn.pack(side="right", padx=(10, 0))
        
        apply_btn = ttk.Button(
            button_frame,
            text="Apply",
            command=self._on_apply_click,
            style='Apply.TButton',
            cursor="hand2"
        )
        apply_btn.pack(side="right")
    
    def _on_apply_click(self):
        """Handle Apply button click."""
        self._apply_settings()
        self.applied = True
        
        # Close dialog
        self.on_close()
        self.dialog.destroy()
    
    def _apply_settings(self):
        """Collect and apply all settings."""
        # Collect all settings
        meeting = VadSettings(
            silence_threshold_db=self.meeting_silence_var.get(),
            speech_threshold_db=self.meeting_speech_var.get(),
            min_silence_sec=self.meeting_min_silence_var.get()
        )
        
        mic = VadSettings(
            silence_threshold_db=self.mic_silence_var.get(),
            speech_threshold_db=self.mic_speech_var.get(),
            min_silence_sec=self.mic_min_silence_var.get()
        )
        
        ui = UiSettings(
            window_opacity=self.ui_opacity_var.get()
        )
        
        self.result = AllSettings(meeting=meeting, mic=mic, ui=ui)
        
        # Call the callback
        self.on_apply(self.result)
    
    def _on_cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.on_close()
        self.dialog.destroy()
    
    def _on_focus_out(self, event):
        """Auto-apply settings when losing focus to main window."""
        # Check if focus went to parent window
        if not self.applied and event.widget == self.dialog:
            # Small delay to check if focus really went to parent
            self.dialog.after(100, self._check_and_auto_apply)
    
    def _check_and_auto_apply(self):
        """Check if dialog still exists and auto-apply if focus is on parent."""
        try:
            if self.dialog.winfo_exists() and not self.applied:
                focused = self.dialog.focus_get()
                # If no widget in dialog has focus, user clicked outside
                if focused is None or focused == self.parent:
                    print("[SETTINGS] Auto-applying on focus loss...")
                    self._apply_settings()
                    self.applied = True
                    self.on_close()
                    self.dialog.destroy()
        except:
            pass
    
    def _on_opacity_change(self, *args):
        """Apply opacity change in real-time and keep dialog on top."""
        try:
            if self.dialog.winfo_exists() and not self.applied:
                new_opacity = self.ui_opacity_var.get()
                # Apply opacity immediately to parent window
                self.parent.attributes("-alpha", new_opacity)
                self.last_opacity = new_opacity
                
                # Aggressively keep dialog on top
                self.dialog.lift()
                self.dialog.attributes("-topmost", True)
                self.dialog.focus_force()
        except Exception as e:
            print(f"[SETTINGS] Error applying opacity: {e}")
