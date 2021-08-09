# Software Name : diSpeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange Labs
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: Olivier Zhang

import os
import json
import logging
import soundfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Iterable
from itertools import product
from utils import synth, get_spectrogram, load_vowel_formants
from tqdm.contrib.concurrent import process_map


class DiSpeech:

    DATASETS_PATH = 'datasets'

    def __init__(self,
                 name: str,
                 factors_kv: dict=None,
                 overwrite: bool=False) -> None:
        """DiSpeech constructor

        Args:
            name (str): name of dataset to generate,
                or name of existring one to load
            factors_kv (dict, optional): factors / array_like of 
                possible values mapping. Defaults to None.
            overwrite (bool, optional): when factors_kv passed,
                has to be True to overwrite existing dataset.
                Else, raises FileExistsError. Defaults to False.
        """
        self.name = name
        self._data = None

        # Defining paths
        self.dataset_path = os.path.join(self.DATASETS_PATH, self.name)
        self.data_path = os.path.join(self.dataset_path, 'data')
        self.metadata_path = os.path.join(self.dataset_path, 'metadata')
        self.waveforms_path = os.path.join(self.dataset_path, 'waveforms')

        if os.path.exists(self.dataset_path) and not overwrite:
            self._load_constructor()
        else:
            self._base_constructor(factors_kv)
        
        self.factor_names = list(self.factors_kv.keys())
        self.factor_sizes = [len(values) for values in self.factors_kv.values()]
        self.factor_space_size = np.prod(self.factor_sizes)

    def _create_paths(self) -> None:
        """Create dataset paths
        """
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
        os.makedirs(self.waveforms_path, exist_ok=True)

    def _base_constructor(self, factors_kv: dict) -> None:
        """Base constructor from factors_kv

        Args:
            factors_kv (dict): factor / values mapping
        """
        if factors_kv is None:
            raise ValueError('factors_kv has to be specified.')
        self.factors_kv = factors_kv
        self._create_paths()
        self._save_metadata()
    
    def _load_constructor(self) -> None:
        """Loading factors_kv.json
        """
        logging.info('Loading factors_kv.json...')
        path = os.path.join(self.metadata_path, 'factors_kv.json')
        with open(path, 'r') as f:
            self.factors_kv = json.load(f)
        logging.info('factors_kv loaded')

    @property
    def factor_space(self) -> product:
        """Returns a generator of factors in dataset's factor space

        Returns:
            product: generator of list of factors
        """
        return product(*self.factors_kv.values())

    @property
    def data(self) -> np.ndarray:      
        """Lazy loading of spectrograms array

        Returns:
            np.ndarray: spectrograms numpy ndarray
                of shape (factor_space_size, 64, 64, 1)
        """
        if self._data is None:
            path = os.path.join(self.data_path, f'{self.name}.npz')
            logging.info('Loading data array...')
            self._data = np.load(path)['spectrograms']
            logging.info('data array loaded.')
        return self._data

    def _save_metadata(self) -> None:
        """Processing and saving factors_kv.
        """        
        self.factors_kv = {k:list(v) for k,v in self.factors_kv.items()}
        path = os.path.join(self.metadata_path, 'factors_kv.json')
        with open(path, 'w') as f:
            json.dump(self.factors_kv, f)

    def _launch_waveform_synthesis(self,
                                   param_values: dict,
                                   file_name: str) -> None:
        """Launchs waveform synthesis from given 'param_values' mapping.

        Args:
            param_values (dict): parameter names / values mapping
            file_name (str): file path where to store generated waveform
        """
        param_values.update(
            {
                'file_path': os.path.join(self.waveforms_path, file_name),
                'save': True,
                'play': False
            }
        )
        synth(**param_values)

    def _gen_single_waveform(self,
                             param_values: Iterable,
                             overwrite: bool=False) -> None:
        """Generate parameters key value mapping for waveform synthesis.

        Args:
            param_values (Iterable): iterable of factor values
                to use for synthesis.
            overwrite (bool, optional): whether to overwrite or
                not existing waveform. Defaults to False.
        """
        param_values = {k:v for k,v in zip(self.factor_names, param_values)}
        file_name = '_'.join(map(str, list(param_values.values()))) + '.wav'
        path = os.path.join(self.waveforms_path, file_name)
        if not overwrite and os.path.exists(path):
            logging.info(f'{file_name} already existing : skipped')
        else:
            self._launch_waveform_synthesis(param_values, file_name)

    def gen_waveforms(self, overwrite: bool=False) -> None:
        """Generate and save waveforms based on factors_kv attribute.

        Args:
            overwrite (bool, optional): whether to overwrite
                existing wav files or not. Defaults to False.
        """
        for param_values in tqdm(
            self.factor_space,
            total=self.factor_space_size,
            desc='Generating waveforms'
        ):
            self._gen_single_waveform(param_values, overwrite)

    def gen_waveforms_multiprocess(self,
                                   overwrite: bool=False,
                                   max_workers: int=6) -> None:        
        """Generate and save waveforms based on factors_kv attribute.
        Synthesis parallelized with 'max_workers' processes.

        Args:
            overwrite (bool, optional): whether to overwrite
                existing wav files or not. Defaults to False.
            max_workers (int, optional): nb workers for parallel synthesis.
                Defaults to 6.
        """
        def _yield_overwrite() -> Iterable[bool]:            
            for _ in range(self.factor_space_size):
                yield overwrite

        process_map(
            self._gen_single_waveform,
            self.factor_space,
            _yield_overwrite(),
            max_workers=max_workers,
            desc='Generating waveforms',
            total=self.factor_space_size
        )

    def gen_spec(self) -> None:
        """Generate log mel filter banks from waveforms
        based on factors_kv attribute, stored in 'data' class attribute
        """
        data = np.ndarray(shape=(self.factor_space_size, 64, 64, 1))
        for i, param_values in tqdm(
            enumerate(self.factor_space),
            total=self.factor_space_size,
            desc='Generating spectrograms'
        ):
            file_name = '_'.join(map(str, param_values)) + '.wav'
            wav_path = os.path.join(self.waveforms_path, file_name)
            array, _ = soundfile.read(wav_path)
            data[i] = get_spectrogram(array)
        
        data_min = np.min(data)
        data = (data - np.min(data))
        data_max = np.max(data)
        data = data / np.max(data)
        scale_values_path = os.path.join(
            self.metadata_path,
            'scale_values.json'
        )
        with open(scale_values_path, 'w') as f:
            json.dump({'min': data_min, 'max': data_max}, f)
        path = os.path.join(self.data_path, f'{self.name}.npz')
        np.savez(path, spectrograms=data)
        self._data = data


class DownSampleDiSpeech(DiSpeech):

    def __init__(
        self,
        name: str,
        base_dataset_name: str=None,
        stride_dict: dict=None,
        overwrite: bool=False
    ) -> None:
        """DownSampleDiSpeech constructor

        Args:
            name (str): dataset name
            base_dataset_name (str): base dataset name to downsample.
                Defaults to None.
            stride_dict (dict): dict of str:int specifying the striding to use 
                to downsample factors each factor. Defaults to None.
            overwrite (bool, optional): whether to overwrite
                existing dataset or not. Defaults to False.
        """
        self.base_dataset_name = base_dataset_name
        self.stride_dict = stride_dict
        super().__init__(name, overwrite=overwrite)

    def _base_constructor(self, *args, **kwargs) -> None:        
        """Generate dataset based on another (base_dataset)
        by downsampling specified factors
        """
        base_dataset = DiSpeech(self.base_dataset_name)
        self.waveforms_path = base_dataset.waveforms_path
        self._create_paths()

        # Striding over base dataset factor values
        factors_kv = base_dataset.factors_kv.copy()
        self.factor_indices = {
            k:range(len(v)) for k,v in base_dataset.factors_kv.items()
        }
        for factor, stride in self.stride_dict.items():
            values = factors_kv[factor]
            s = slice(0, len(values), stride)
            factors_kv[factor] = values[s]
            self.factor_indices[factor] = self.factor_indices[factor][s]
        
        # Creating index to downsample data array of spectrograms
        cumprod = np.cumprod(base_dataset.factor_sizes)
        factor_bases = base_dataset.factor_space_size / cumprod
        index = np.array(
            np.dot(
                [i for i in product(*self.factor_indices.values())],
                factor_bases
            ),
            dtype=np.int64
        )

        # Creating and saving data array of spectrograms
        self._data = base_dataset.data[index]
        path = os.path.join(self.data_path, f'{self.name}.npz')
        np.savez(path, spectrograms=self.data)

        super()._base_constructor(factors_kv)

    def _load_constructor(self) -> None:
        """Loading factor_indices, stride_dict and base_dataset_name
        """        
        logging.info('Loading factor_indices.json...')
        path = os.path.join(self.metadata_path, 'factor_indices.json')
        with open(path, 'r') as f:
            self.factor_indices = json.load(f)
        logging.info('factor_indices loaded')
        logging.info('Loading stride_dict.json...')
        path = os.path.join(self.metadata_path, 'stride_dict.json')
        with open(path, 'r') as f:
            self.stride_dict = json.load(f)
        logging.info('Done')
        logging.info('Loading base_dataset_name.txt...')
        path = os.path.join(self.metadata_path, 'base_dataset_name.txt')
        with open(path, 'r') as f:
            self.base_dataset_name = f.read()
        logging.info('Done')
        super()._load_constructor()

    def _save_metadata(self) -> None:
        """Save relative factor indices prior basic class saving method,
        stride_dict and base dataset name
        """
        # Saving factor_indicies
        self.factor_indices = {k:list(v) for k,v in self.factor_indices.items()}
        path = os.path.join(self.metadata_path, 'factor_indices.json')
        with open(path, 'w') as f:
            json.dump(self.factor_indices, f)
        # Saving stride_dict
        path = os.path.join(self.metadata_path, 'stride_dict.json')
        with open(path, 'w') as f:
            json.dump(self.stride_dict, f)
        # Saving base dataset name
        path = os.path.join(self.metadata_path, 'base_dataset_name.txt')
        with open(path, 'w') as f:
            f.write(self.base_dataset_name)
        super()._save_metadata()
    
    def gen_waveforms(self, *args, **kwargs) -> None:
        """No waveform generation needed
        """
        logging.info(
            f'Waveforms downsampled from {self.base_dataset_name}, \
                no need to generate them.'
        )

    def gen_waveforms_multiprocess(self, *args, **kwargs) -> None:
        """No waveform generation needed
        """
        logging.info(
            f'Spectrogram downsampled from {self.base_dataset_name}, \
                no need to generate them.'
        )

    def gen_spec(self, *args, **kwargs) -> None:
        """No spectrogram generation needed
        """
        logging.info(
            f'Spectrogram downsampled from {self.base_dataset_name}, \
                no need to generate them.'
        )


class VowelDiSpeech(DiSpeech):
    
    def __init__(self,
                 name: str,
                 factors_kv: dict=None,
                 overwrite: bool=False) -> None:
        """Adding vowel formants DataFrame to class attributes
        """
        self.vowel_formants = load_vowel_formants()[['F1', 'F2', 'F3']]
        super().__init__(name, factors_kv=factors_kv, overwrite=overwrite)

    def _save_metadata(self) -> None:
        """Adding vowels as generative factor before saving factors_kv.
        """
        self.factors_kv.update({'vowel' : self.vowel_formants.index.tolist()})
        super()._save_metadata()

    def _launch_waveform_synthesis(self,
                                   param_values: dict,
                                   file_name: str) -> None:
        """Redefining _launch_waveform_synthesis method to replace vowel factor
        with corresponding formants prior default processing.

        Args:
            param_values (dict): parameter names / values mapping
            file_name (str): file path where to store generated waveform
        """
        vowel = param_values.pop('vowel')
        param_values.update(self.vowel_formants.loc[vowel].to_dict())
        return super()._launch_waveform_synthesis(param_values, file_name)


class FreeDiSpeech(DiSpeech):

    def __init__(
        self, name: str,
        factors_df: pd.DataFrame=None,
        overwrite: bool=False
    ) -> None:
        self.name = name
        self.factors_df = factors_df
        self._data = None

        # Defining paths
        self.dataset_path = os.path.join(self.DATASETS_PATH, self.name)
        self.data_path = os.path.join(self.dataset_path, 'data')
        self.metadata_path = os.path.join(self.dataset_path, 'metadata')
        self.waveforms_path = os.path.join(self.dataset_path, 'waveforms')

        if os.path.exists(self.dataset_path) and not overwrite:
            self._load_constructor()
        else:
            self._base_constructor()


        self.factor_names = list(self.factors_df.columns)
        self.factor_space_size = self.factors_df.shape[0]

    def _base_constructor(self) -> None:
        self._create_paths()
        self._save_metadata()
    
    def _save_metadata(self) -> None:
        self.factors_df.to_csv(os.path.join(self.metadata_path, 'factors.csv'))
    
    def _load_constructor(self) -> None:
        path = os.path.join(self.metadata_path, 'factors.csv')
        logging.info(f'Loading {path}')
        self.factors_df = pd.read_csv(path, index_col=0)
        logging.info('Done')

    @property
    def factor_space(self) -> np.ndarray:
        return self.factors_df.to_numpy()
