# Ready-dev-one-AI
AI asistan ,  Integrate AI to 

Overlay assistant for interviews:
- Captures meeting/system audio via BlackHole
- Detects pauses and saves WAV segments
- Modular STT layer (OpenAI or local)


## Requirements
- macOS
- Python 3.10+
- BlackHole 2ch installed


brew install python-tk

### Getting Started

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
cp .env.example .env

### macOS audio routing (critical)

Goal: route system audio (YouTube/Teams/Meet) into BlackHole so Python can read it.

1 Open Audio MIDI Setup

2 Click + → Create Multi-Output Device

3 Check:

- Your Speakers / Headphones
- BlackHole 2ch

4 Enable Drift Correction for BlackHole 2ch

5 System Settings → Sound → Output → select Multi-Output Device

Now BlackHole will receive whatever is playing.

### Verify audio is flowing

```bash
python scripts/list_devices.py
python scripts/monitor_input_level.py
```

### Run the app

```bash
python run.py
```