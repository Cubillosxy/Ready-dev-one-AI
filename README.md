# Ready-dev-one-AI

AI Assistant with real-time voice transcription and intelligent suggestions.

## Features

- **Bilingual Support**: Toggle between English and Spanish with a single button
- **Dual Audio Input Modes**:
  - **Meeting Mode**: Captures system audio (Chrome/Teams/Meet) via BlackHole
  - **Mic Mode**: Captures microphone input directly
- **Real-time Streaming Transcription**: Using Vosk for local STT processing
- **Voice Activity Detection (VAD)**: Automatically detects speech segments and pauses
- **AI Assistant Integration**: Sends transcribed text to OpenAI LLM for intelligent suggestions
- **Overlay UI**: Always-on-top draggable window with real-time status

## Requirements

- macOS
- Python 3.10+
- BlackHole 2ch (for meeting mode only)
- Vosk model files (downloaded automatically or manually)

## Installation

### 1. Install Python Tkinter

```bash
brew install python-tk
```

### 2. Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

### 4. macOS Audio Routing (for Meeting Mode)

**Goal**: Route system audio (Chrome/Teams/Meet) into BlackHole so the app can capture it.

1. Open **Audio MIDI Setup** (Applications → Utilities)
2. Click **+** → **Create Multi-Output Device**
3. Check:
   - Your Speakers / Headphones
   - BlackHole 2ch
4. Enable **Drift Correction** for BlackHole 2ch
5. Go to **System Settings** → **Sound** → **Output** → Select **Multi-Output Device**

Now BlackHole will receive whatever is playing on your system.

### 5. Verify Audio Setup

```bash
python scripts/list_devices.py          # List all audio devices
python scripts/monitor_input_level.py   # Monitor input levels
```

## Usage

### Run the Application

```bash
python run.py
```

### UI Controls

The overlay window provides the following buttons:

- **Toggle Listening**: Enable/disable audio capture and transcription
- **Answer Now**: Manually finalize current speech segment
- **Lang: EN/ES**: Switch between English and Spanish language models
- **Input: Meet/Mic**: Toggle between Meeting (BlackHole) and Microphone input

**Keyboard Shortcuts**:
- **ESC**: Quit the application
- **Drag**: Click and drag anywhere on the window to reposition

### Display Information

The UI shows:
- **Input device**: Current audio input device
- **Listening**: ON/OFF status
- **State**: Current VAD state (silence/speech)
- **Level**: Audio level in dB
- **Last pause**: Duration of last silence period
- **Last capture**: Filename and duration of last saved segment
- **Partial**: Real-time partial transcription
- **Final**: Finalized transcription text
- **Suggestion**: AI assistant response

## Configuration

Edit `rdoai/config.py` or your `.env` file to customize:

- **Audio settings**: Sample rate, channels, device indices
- **VAD thresholds**: Silence/speech detection levels
- **STT provider**: OpenAI or local Vosk
- **LLM settings**: Model, API parameters
- **UI appearance**: Window opacity, positioning

## How It Works

1. **Audio Capture**: Captures audio from selected input (mic or system audio)
2. **Streaming STT**: Processes audio in real-time using Vosk, providing partial and final transcriptions
3. **Segment Detection**: VAD detects speech pauses and saves segments as WAV files
4. **LLM Processing**: Sends finalized transcriptions to OpenAI for intelligent suggestions
5. **UI Updates**: Displays real-time transcription and AI responses in overlay window

## Troubleshooting

**No audio levels showing?**
- Run `python scripts/list_devices.py` to verify device indices
- Check that BlackHole is installed and Multi-Output Device is selected
- Verify microphone permissions in System Settings → Privacy & Security

**Transcription not working?**
- Ensure Vosk model files are downloaded
- Check STT provider configuration in `.env`
- Verify language setting matches your speech

**LLM not responding?**
- Verify OpenAI API key in `.env`
- Check console for error messages
- Ensure internet connection is active

## Testing

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=rdoai --cov-report=term-missing

# Run specific test file
pytest tests/test_config.py -v

# Run tests in a specific directory
pytest tests/audio/ -v
```

### Test Structure

The test suite mirrors the source code structure:

```
tests/
├── test_config.py          # Configuration tests
├── audio/
│   └── test_segmenter.py   # Audio segmentation tests
├── pipeline/
│   └── test_events.py      # Event dataclass tests
├── stt/
│   └── test_streaming_base.py  # STT base class tests
└── utils/                  # Utility function tests
```

## Project Structure

```
rdoai/
├── app/          # Main application controller
├── audio/        # Audio capture and VAD
├── stt/          # Speech-to-text (OpenAI, Vosk)
├── llm/          # LLM integration
├── pipeline/     # Worker threads and event handling
├── ui/           # Tkinter overlay window
└── config.py     # Configuration dataclasses
```