import time
import numpy as np
import sounddevice as sd

DEVICE_INDEX = 1
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

def rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    rms = max(rms, 1e-8)
    return 20.0 * np.log10(rms)

def main():
    dev = sd.query_devices(DEVICE_INDEX)
    print(f"Using device #{DEVICE_INDEX}: {dev['name']}")
    print("Play YouTube/Meet audio now. You should see levels change from ~-160 dB.")

    last_print = 0.0

    def cb(indata, frames, time_info, status):
        nonlocal last_print
        x = indata[:, 0].astype(np.float32)
        db = rms_db(x)

        now = time.time()
        if now - last_print >= 0.5:
            last_print = now
            print(f"level={db:6.1f} dB frames={frames}")

    with sd.InputStream(device=DEVICE_INDEX, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, dtype="float32", callback=cb):
        while True:
            time.sleep(0.2)

if __name__ == "__main__":
    main()
