from dataset import DownSampleDiSpeech


base_dataset = 'diSpeech-Bbig'
stride_dict = {'F0': 3}
dispeech_f0 = DownSampleDiSpeech('diSpeech-F0', base_dataset, stride_dict)
