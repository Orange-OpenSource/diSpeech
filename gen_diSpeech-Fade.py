# Software Name : diSpeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange Labs
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: Olivier Zhang

from dataset import DownSampleDiSpeech


base_dataset = 'diSpeech-Bbig'
stride_dict = {'fade': 3}
dispeech_fade = DownSampleDiSpeech('diSpeech-Fade', base_dataset, stride_dict)
