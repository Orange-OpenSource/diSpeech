from dataset import DownSampleDiSpeech


base_dataset = 'diSpeech-Bbig'
stride_dict = {'fade': 3}
dispeech_fade = DownSampleDiSpeech('diSpeech-Fade', base_dataset, stride_dict)
