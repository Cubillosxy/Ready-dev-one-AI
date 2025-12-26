import numpy as np

def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -160.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    rms = max(rms, 1e-8)
    return 20.0 * np.log10(rms)
