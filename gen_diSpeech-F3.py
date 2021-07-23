from dataset import DownSampleDiSpeech


base_dataset = 'diSpeech-Bbig'
stride_dict = {'F3': 3}
dispeech_f3 = DownSampleDiSpeech('diSpeech-F3', base_dataset, stride_dict)
