"""Tests for OverlayWindow."""
import pytest
from unittest.mock import MagicMock, patch
from rdoai.ui.overlay import OverlayWindow, OverlayState


@pytest.fixture
def mock_tk(mocker):
    return mocker.patch("rdoai.ui.overlay.tk")


class TestOverlayWindow:
    def test_init(self, mock_tk):
        # Setup mocks for StringVar
        mock_tk.StringVar.return_value = MagicMock()
        
        window = OverlayWindow(geometry="100x100", alpha=0.5, always_on_top=True)
        
        mock_tk.Tk.assert_called_once()
        window.root.overrideredirect.assert_called_with(True)
        window.root.attributes.assert_any_call("-alpha", 0.5)
        window.root.attributes.assert_any_call("-topmost", True)
        window.root.geometry.assert_called_with("100x100")

    def test_render_updates_vars(self, mock_tk):
        # Create unique mocks for each StringVar
        vars_mocks = {}
        def stringvar_side_effect(value=None):
            m = MagicMock()
            m.get.return_value = value
            return m
        
        mock_tk.StringVar.side_effect = stringvar_side_effect
        
        window = OverlayWindow("1x1", 1.0, False)
        
        state = OverlayState(
            listening=True,
            level_db=-20.5,
            vad_state="speech",
            last_pause_sec=1.23,
            last_capture="test.wav",
            device_label="Mic 1",
            hint="Ready",
            transcript_partial="hello",
            transcript_final="Hello world",
            suggestion="Use Python"
        )
        
        window.render(state)
        
        # Check some variables updated
        window.status_var.set.assert_called_with("AI: ON")
        window.level_var.set.assert_called_with("Level: -20.5 dB")
        window.suggestion_var.set.assert_called_with("Suggestion: Use Python")
        window.partial_var.set.assert_called_with("Partial: hello")

    def test_set_handlers(self, mock_tk):
        window = OverlayWindow("1x1", 1.0, False)
        
        h1 = MagicMock()
        h2 = MagicMock()
        window.set_handlers(h1, h2, MagicMock(), MagicMock())
        
        assert window._on_toggle_listening == h1
        assert window._on_finalize == h2

    def test_every(self, mock_tk):
        window = OverlayWindow("1x1", 1.0, False)
        fn = MagicMock()
        window.every(100, fn)
        window.root.after.assert_called_with(100, fn)
