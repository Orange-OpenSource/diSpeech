import numpy as np
from dataset import DiSpeech
from utils import load_vowel_formants


N = 5

vf = load_vowel_formants()
factors_kv = {
    'F1': np.linspace(min(vf['F1']), max(vf['F1']), N, dtype=int).tolist(),
    'F2': np.linspace(min(vf['F2']), max(N['F2']), N, dtype=int).tolist(),
    'F3': np.linspace(min(vf['F3']), max(N['F3']), N, dtype=int).tolist(),
    'F0': np.linspace(50, 200, N, dtype=int).tolist(),
    'fade': np.linspace(0, 99, N, dtype=int).tolist()
}
dispeech_bsmall = DiSpeech('diSpeech-Bsmall', factors_kv)
dispeech_bsmall.gen_waveforms_multiprocess()
dispeech_bsmall.gen_spec()