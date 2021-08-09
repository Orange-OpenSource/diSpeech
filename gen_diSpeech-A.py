# Software Name : diSpeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange Labs
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: Olivier Zhang

import numpy as np
from dataset import DiSpeech
from utils import load_vowel_formants


N_F = 25
N_F0 = 4
N_FADE = 10

vf = load_vowel_formants()
factors_kv = {
    'F1': np.linspace(min(vf['F1']), max(vf['F1']), N_F, dtype=int).tolist(),
    'F2': np.linspace(min(vf['F2']), max(vf['F2']), N_F, dtype=int).tolist(),
    'F3': np.linspace(min(vf['F3']), max(vf['F3']), N_F, dtype=int).tolist(),
    'F0': np.linspace(50, 200, N_F0, dtype=int).tolist(),
    'fade': np.linspace(0, 99, N_FADE, dtype=int).tolist()
}
dispeech_a = DiSpeech('diSpeech-A', factors_kv)
dispeech_a.gen_waveforms_multiprocess()
dispeech_a.gen_spec()
