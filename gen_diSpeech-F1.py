from dataset import DownSampleDiSpeech


base_dataset = 'diSpeech-Bbig'
stride_dict = {'F1': 3}
dispeech_f1 = DownSampleDiSpeech('diSpeech-F1', base_dataset, stride_dict)
