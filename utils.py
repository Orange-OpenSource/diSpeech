import torch
import tdklatt
import numpy as np
import pandas as pd
from scipy import stats
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank


def load_vowel_formants(fn: str='./vowel_formants.csv') -> pd.DataFrame:
    return pd.read_csv(fn, index_col=0)

def synth(F0,
          F1,
          F2,
          F3,
          AV=60,
          fade=50,
          play=True,
          save=False,
          file_path='sample.wav') -> np.ndarray:
    s = tdklatt.klatt_make(tdklatt.KlattParam1980(DUR=1))
    N = s.params["N_SAMP"]
    
#     s.params['F0'][:] = F0
    s.params['F0'] = np.linspace(F0, F0 * (1 - fade / 100), N)
    s.params['AV'][:] = AV
#     s.params['AV'] = np.linspace(1, 0, N) ** 0.1 * AV
    FF = np.asarray(s.params['FF'])
    FF[:3, :] = np.outer(np.ones(N), np.array([F1, F2, F3])).T
    s.params['FF'] = FF
    
    s.run()
    if play:
        s.play()
    if save:
        s.save(file_path)
    return s._get_int16at16K()

def get_spectrogram(array, hop_length=252):
    extractor = LogMelFbank(
        fs=16000,
        fmin=80,
        fmax=7600,
        n_mels=64,
        hop_length=hop_length,
        n_fft=1024,
        win_length=None
    )
    t = torch.Tensor(array).unsqueeze(0)
    l = torch.Tensor([len(array)])
    features, _ = extractor(t, l)
    return np.expand_dims(np.rot90(features.numpy(), axes=(1, 2)), 3)
