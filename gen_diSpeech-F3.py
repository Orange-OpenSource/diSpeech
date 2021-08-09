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
stride_dict = {'F3': 3}
dispeech_f3 = DownSampleDiSpeech('diSpeech-F3', base_dataset, stride_dict)
