from dataset import DownSampleDiSpeech


base_dataset = 'diSpeech-Bbig'
stride_dict = {'F2': 3}
dispeech_f2 = DownSampleDiSpeech('diSpeech-F2', base_dataset, stride_dict)
