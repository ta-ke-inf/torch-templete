import torchaudio
import os

from const.path import (
    DATA_PATH
)
if __name__ == "__main__":
    SAMPLE_WAV_PATH = os.path.join(DATA_PATH, "X097_DE.wav")
    waveform, origin_fs = torchaudio.load(filepath=SAMPLE_WAV_PATH)
    print(waveform)
