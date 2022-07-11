# Software Name : diSpeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: Olivier Zhang

import numpy as np
from dataset import DiSpeech
from utils import load_vowel_formants


N = 15

vf = load_vowel_formants()
factors_kv = {
    'F1': np.linspace(min(vf['F1']), max(vf['F1']), N, dtype=int).tolist(),
    'F2': np.linspace(min(vf['F2']), max(vf['F2']), N, dtype=int).tolist(),
    'F3': [3000],
    'F0': np.linspace(70, 250, N, dtype=int).tolist(),
    'fade': [0]
}
dispeech_a = DiSpeech('diSpeech_F3_fixed', factors_kv)
dispeech_a.gen_waveforms_multiprocess()
dispeech_a.gen_spec()
