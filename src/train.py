import torchaudio
import torchaudio.functional as F
import numpy as np
import torch
import torchaudio.transforms as T
from torchvision import transforms


import os

from const.path import (
    DATA_PATH
)
if __name__ == "__main__":
    SAMPLE_WAV_PATH = os.path.join(DATA_PATH, "X097_DE.wav")
    waveform, sample_rate = torchaudio.load(filepath=SAMPLE_WAV_PATH)
    #start_point = sample_rate * 5
    #cut_width = sample_rate * 10
    #waveform = waveform[:, start_point : (start_point + cut_width)]
    waveform = waveform[:, 0 : sample_rate * 60]
    #-----
    #new_sample_rate = sample_rate / 2
    #channel=0
    #waveform_transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1))
    print(waveform)
    #-----

    # Melspectrogram の設定
    sample_rate=sample_rate
    n_mels = 224#128
    n_fft = 4096#2150 #2048
    win_length = None
    hop_length = n_fft//2 #4
    window_fn = torch.hann_window

    # Melspectrogram の計算
    spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=window_fn,
        power=2.0,
    )
    spec = spectrogram(waveform)
    print(spec.shape)
    #spec = spectrogram(waveform)
    #spec_aug = torch.nn.Sequential(
    #    T.TimeMasking(time_mask_param=30, p=0.1),
    #    T.FrequencyMasking(freq_mask_param=30),
    #)
    #spec = spec_aug(spec)
    #print(type(spec))
    # 縦軸をデシベルに変換

    transform = transforms.TenCrop(size=(224, 224))
    spec_list = transform(spec)
    spec_db_list = []
    print(spec_list[0].shape)
    for i in range(10):
        spec_db_list.append(librosa.power_to_db(spec_list[i][0]))#spec[0])
        print(spec_db_list[i].shape)
    plt.imshow(spec_db_list[9], origin="lower")
    #plt.savefig("1.png", bbox_inches='tight', pad_inches=0)
