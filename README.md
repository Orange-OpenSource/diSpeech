# diSpeech
Materials to generate diSpeech datasets for speech disentanglement purposes. Klatt synthetizer is used to generate phonemes.

## Files
- `dataset.py`: main classes to use to generate and load diSpeech datasets.
- `utils.py`: usefull functions to synthetize phoneme waveforms and compute log mel-filter bank from waveforms.
- `gen_*.py`: scripts to generate mentioned datasets in *diSpeech : A synthetic toy dataset for speech disentangling* (Table 2).
- `tdklatt.py`: Klatt synthetizer implementation, from [https://github.com/guestdaniel/tdklatt](https://github.com/guestdaniel/tdklatt).
- `vowel_formants.csv`: reference formants of french vowels.