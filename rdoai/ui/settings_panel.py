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


class SettingsPanel(tk.Frame):
    """Dropdown settings panel embedded in the overlay window."""
    
    def __init__(self, parent: tk.Widget, current_settings: AllSettings, on_change: Callable[[AllSettings], None]):
        super().__init__(parent, bg="#1e1e1e")
        self.on_change = on_change
        self.current_settings = current_settings
        
        # Configure ttk styles for close button
        self.style = ttk.Style()
        self.style.theme_use('default')
        
        # Custom style for Close button
        self.style.configure(
            'Close.TButton',
            background='#f44336',
            foreground='white',
            borderwidth=0,
            focuscolor='none',
            font=('Helvetica', 9),
            padding=(10, 4)
        )
        self.style.map(
            'Close.TButton',
            background=[('active', '#da190b')],
            foreground=[('active', 'white')]
        )
        
        # Header with title and close button
        header = tk.Frame(self, bg="#1e1e1e")
        header.pack(fill="x", padx=15, pady=(8, 5))
        
        title = tk.Label(
            header,
            text="Settings",
            font=("Helvetica", 11, "bold"),
            fg="white",
            bg="#1e1e1e"
        )
        title.pack(side="left")
        
        close_btn = ttk.Button(
            header,
            text="Close",
            command=self.hide,
            style='Close.TButton',
            cursor="hand2"
        )
        close_btn.pack(side="right")
        
        # Create sections
        self._create_meeting_section(current_settings.meeting)
        self._create_separator()
        self._create_mic_section(current_settings.mic)
        self._create_separator()
        self._create_ui_section(current_settings.ui)
        
        # Initially hidden
        self.is_visible = False
        
    def _create_meeting_section(self, settings: VadSettings):
        """Create Meeting mode settings section."""
        frame = tk.Frame(self, bg="#1e1e1e")
        frame.pack(fill="x", padx=15, pady=(3, 2))
        
        label = tk.Label(
            frame,
            text="Meeting Mode",
            font=("Helvetica", 9, "bold"),
            fg="#4CAF50",
            bg="#1e1e1e"
        )
        label.pack(anchor="w", pady=(0, 3))
        
        self.meeting_silence_var = tk.DoubleVar(value=settings.silence_threshold_db)
        self.meeting_speech_var = tk.DoubleVar(value=settings.speech_threshold_db)
        self.meeting_min_silence_var = tk.DoubleVar(value=settings.min_silence_sec)
        
        # Add trace to apply changes immediately
        self.meeting_silence_var.trace_add("write", self._on_setting_change)
        self.meeting_speech_var.trace_add("write", self._on_setting_change)
        self.meeting_min_silence_var.trace_add("write", self._on_setting_change)
        
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
        frame = tk.Frame(self, bg="#1e1e1e")
        frame.pack(fill="x", padx=15, pady=(3, 2))
        
        label = tk.Label(
            frame,
            text="Mic Mode",
            font=("Helvetica", 9, "bold"),
            fg="#2196F3",
            bg="#1e1e1e"
        )
        label.pack(anchor="w", pady=(0, 3))
        
        self.mic_silence_var = tk.DoubleVar(value=settings.silence_threshold_db)
        self.mic_speech_var = tk.DoubleVar(value=settings.speech_threshold_db)
        self.mic_min_silence_var = tk.DoubleVar(value=settings.min_silence_sec)
        
        # Add trace to apply changes immediately
        self.mic_silence_var.trace_add("write", self._on_setting_change)
        self.mic_speech_var.trace_add("write", self._on_setting_change)
        self.mic_min_silence_var.trace_add("write", self._on_setting_change)
        
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
        frame = tk.Frame(self, bg="#1e1e1e")
        frame.pack(fill="x", padx=15, pady=(3, 8))
        
        label = tk.Label(
            frame,
            text="Window Settings",
            font=("Helvetica", 9, "bold"),
            fg="#FF9800",
            bg="#1e1e1e"
        )
        label.pack(anchor="w", pady=(0, 3))
        
        self.ui_opacity_var = tk.DoubleVar(value=settings.window_opacity)
        
        # Add trace to apply changes immediately
        self.ui_opacity_var.trace_add("write", self._on_setting_change)
        
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
        container.pack(fill="x", pady=2)
        
        # Label and value display
        label_frame = tk.Frame(container, bg="#1e1e1e")
        label_frame.pack(fill="x")
        
        label = tk.Label(
            label_frame,
            text=label_text,
            font=("Helvetica", 8),
            fg="white",
            bg="#1e1e1e"
        )
        label.pack(side="left")
        
        value_label = tk.Label(
            label_frame,
            textvariable=variable,
            font=("Helvetica", 8, "bold"),
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
            activebackground="#4CAF50",
            length=200
        )
        slider.pack(fill="x", pady=(1, 0))
    
    def _create_separator(self):
        """Create a visual separator between sections."""
        sep = tk.Frame(self, bg="#424242", height=1)
        sep.pack(fill="x", padx=15, pady=5)
    
    def _on_setting_change(self, *args):
        """Apply settings change immediately."""
        try:
            # Collect all current settings
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
            
            self.current_settings = AllSettings(meeting=meeting, mic=mic, ui=ui)
            
            # Call the callback to apply changes
            self.on_change(self.current_settings)
        except Exception as e:
            print(f"[SETTINGS PANEL] Error applying change: {e}")
    
    def show(self):
        """Show the settings panel."""
        if not self.is_visible:
            self.pack(fill="x", after=self.master.winfo_children()[0])  # Pack after menu bar
            self.is_visible = True
    
    def hide(self):
        """Hide the settings panel."""
        if self.is_visible:
            self.pack_forget()
            self.is_visible = False
    
    def toggle(self):
        """Toggle panel visibility."""
        if self.is_visible:
            self.hide()
        else:
            self.show()
