import wave
import numpy as np

def write_wav_mono_int16(path: str, pcm_float: np.ndarray, sample_rate: int) -> None:
    pcm_int16 = np.clip(pcm_float, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
