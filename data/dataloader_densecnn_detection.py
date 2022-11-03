import sys
sys.path.append('../')
import pyedflib
import utils
from data.data_utils import *
from constants import INCLUDED_CHANNELS, FREQUENCY
from utils import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import math
import h5py
import numpy as np
import os
import pickle
import scipy
import scipy.signal
from pathlib import Path

repo_paths = str(Path.cwd()).split('eeg-gnn-ssl')
repo_paths = Path(repo_paths[0]).joinpath('eeg-gnn-ssl')
sys.path.append(repo_paths)
FILEMARKER_DIR = Path(repo_paths).joinpath('data/file_markers_detection')


def computeSliceMatrix(
        h5_fn,
        edf_fn,
        clip_idx,
        clip_len=60,
        is_fft=False):
    """
    Comvert entire EEG sequence into clips of length clip_len
    Args:
        h5_fn: file name of resampled signal h5 file (full path)
        clip_idx: index of current clip/sliding window
        clip_len: sliding window size or EEG clip length, in seconds, int
        is_fft: whether to perform FFT on raw EEG data
    Returns:
        slices: list of EEG clips, each having shape (clip_len*freq, num_channels, time_step_size*freq)
        seizure_labels: list of seizure labels for each clip, 1 for seizure, 0 for no seizure
    """
    with h5py.File(h5_fn, 'r') as f:
        signal_array = f["resampled_signal"][()]
        resampled_freq = f["resample_freq"][()]
    assert resampled_freq == FREQUENCY

    # get seizure times
    seizure_times = getSeizureTimes(edf_fn.split('.edf')[0])

    # Iterating through signal
    physical_clip_len = int(FREQUENCY * clip_len)

    start_window = clip_idx * physical_clip_len
    end_window = start_window + physical_clip_len
    # (num_channels, physical_clip_len)
    eeg_clip = signal_array[:, start_window:end_window]

    # determine if there's seizure in current clip
    is_seizure = 0
    for t in seizure_times:
        start_t = int(t[0] * FREQUENCY)
        end_t = int(t[1] * FREQUENCY)
        if not ((end_window < start_t) or (start_window > end_t)):
            is_seizure = 1
            break

    if is_fft:
        eeg_clip, _ = computeFFT(eeg_clip, n=physical_clip_len)

    return eeg_clip, is_seizure


def parseTxtFiles(split_type, seizure_file, nonseizure_file,
                  cv_seed=123, scale_ratio=1):

    np.random.seed(cv_seed)

    seizure_str = []
    nonseizure_str = []

    seizure_contents = open(seizure_file, "r")
    seizure_str.extend(seizure_contents.readlines())

    nonseizure_contents = open(nonseizure_file, "r")
    nonseizure_str.extend(nonseizure_contents.readlines())

    # balanced dataset if train
    if split_type == 'train':
        num_dataPoints = int(scale_ratio * len(seizure_str))
        print('number of seizure files: ', num_dataPoints)
        sz_ndxs_all = list(range(len(seizure_str)))
        np.random.shuffle(sz_ndxs_all)
        sz_ndxs = sz_ndxs_all[:num_dataPoints]
        seizure_str = [seizure_str[i] for i in sz_ndxs]
        np.random.shuffle(nonseizure_str)
        nonseizure_str = nonseizure_str[:num_dataPoints]

    combined_str = seizure_str + nonseizure_str

    np.random.shuffle(combined_str)

    combined_tuples = []
    for i in range(len(combined_str)):
        tup = combined_str[i].strip("\n").split(",")
        tup[1] = int(tup[1])
        combined_tuples.append(tup)

    print_str = 'Number of clips in ' + \
        split_type + ': ' + str(len(combined_tuples))
    print(print_str)

    return combined_tuples


class SeizureDataset(Dataset):
    def __init__(
            self,
            input_dir,
            raw_data_dir,
            time_step_size=1,
            max_seq_len=60,
            standardize=True,
            scaler=None,
            split='train',
            data_augment=False,
            sampling_ratio=1,
            seed=123,
            use_fft=False,
            preproc_dir=None):
        """
        Args:
            input_dir: dir to resampled signals h5 files
            raw_data_dir: dir to TUSZ edf files
            time_step_size: int, in seconds
            max_seq_len: int, eeg clip length, in seconds
            standardize: if True, will z-normalize wrt train set
            scaler: scaler object for standardization
            split: train, dev or test
            data_augment: if True, perform random augmentation on EEG
            sampling_ratio: ratio of positive to negative examples for undersampling
            seed: random seed for undersampling
            use_fft: whether perform Fourier transform
            preproc_dir: dir to preprocessed Fourier transformed data, optional 
        """
        if standardize and (scaler is None):
            raise ValueError('To standardize, please provide scaler.')

        self.input_dir = input_dir
        self.raw_data_dir = raw_data_dir
        self.max_seq_len = max_seq_len
        self.standardize = standardize
        self.scaler = scaler
        self.split = split
        self.data_augment = data_augment
        self.use_fft = use_fft
        self.preproc_dir = preproc_dir

        # get full paths to all raw edf files
        self.edf_files = []
        for path, subdirs, files in os.walk(raw_data_dir):
            for name in files:
                if ".edf" in name:
                    self.edf_files.append(os.path.join(path, name))

        seizure_file = os.path.join(
            FILEMARKER_DIR,
            split +
            'Set_seq2seq_' +
            str(max_seq_len) +
            's_sz.txt')
        nonSeizure_file = os.path.join(
            FILEMARKER_DIR,
            split +
            'Set_seq2seq_' +
            str(max_seq_len) +
            's_nosz.txt')
        self.file_tuples = parseTxtFiles(
            split,
            seizure_file,
            nonSeizure_file,
            cv_seed=seed,
            scale_ratio=sampling_ratio)

        self.size = len(self.file_tuples)

        # Get sensor ids
        self.sensor_ids = [x.split(' ')[-1] for x in INCLUDED_CHANNELS]

        targets = []
        for i in range(len(self.file_tuples)):
            if self.file_tuples[i][-1] == 0:
                targets.append(0)
            else:
                targets.append(1)
        self._targets = targets

    def __len__(self):
        return self.size

    def targets(self):
        return self._targets

    def _random_reflect(self, eeg_clip):
        swap_pairs = get_swap_pairs(INCLUDED_CHANNELS)
        eeg_clip_reflect = eeg_clip.copy()
        if(np.random.choice([True, False])):
            for pair in swap_pairs:
                eeg_clip_reflect[:, [pair[0], pair[1]]
                                 ] = eeg_clip[:, [pair[1], pair[0]]]

        return eeg_clip_reflect

    def _random_scale(self, eeg_clip):
        scale_factor = np.random.uniform(0.8, 1.2)
        if self.is_fft:
            eeg_clip += np.log(scale_factor)
        else:
            eeg_clip *= scale_factor
        return eeg_clip

    def __getitem__(self, idx):
        h5_fn, seizure_label = self.file_tuples[idx]
        clip_idx = int(h5_fn.split('_')[-1].split('.h5')[0])

        edf_file = [file for file in self.edf_files if h5_fn.split('.edf')[
            0] + '.edf' in file]
        assert len(edf_file) == 1
        edf_file = edf_file[0]

        # preprocess
        if self.preproc_dir is None:
            resample_sig_dir = os.path.join(
                self.input_dir, h5_fn.split('.edf')[0] + '.h5')
            eeg_clip, is_seizure = computeSliceMatrix(
                h5_fn=resample_sig_dir, edf_fn=edf_file, clip_idx=clip_idx,
                clip_len=self.max_seq_len, is_fft=self.use_fft)
        else:
            with h5py.File(os.path.join(self.preproc_dir, h5_fn), 'r') as hf:
                eeg_clip = hf['clip'][()]
            eeg_clip = np.transpose(eeg_clip, (1, 0, 2)).reshape(19, -1).T # (clip_len*freq, num_channels)

        # data augmentation
        if self.data_augment:
            eeg_clip = self._random_reflect(eeg_clip)
            eeg_clip = self._random_scale(eeg_clip)

        # standardize wrt train mean and std
        if self.standardize:
            eeg_clip = self.scaler.transform(eeg_clip)

        # convert to tensors
        # (max_seq_len, num_nodes, input_dim)
        x = torch.FloatTensor(eeg_clip)
        y = torch.FloatTensor([seizure_label])
        seq_len = torch.LongTensor([self.max_seq_len])
        writeout_fn = h5_fn.split('.h5')[0]

        return (x, y, seq_len, [], [], writeout_fn) # for dataloader consistency


def load_dataset_detection(
        input_dir,
        raw_data_dir,
        train_batch_size,
        test_batch_size=None,
        max_seq_len=60,
        standardize=True,
        num_workers=8,
        augmentation=False,
        use_fft=False,
        sampling_ratio=1,
        seed=123,
        preproc_dir=None):
    """
    Args:
        input_dir: dir to preprocessed h5 file
        raw_data_dir: dir to TUSZ raw edf files
        train_batch_size: int
        test_batch_size: int
        max_seq_len: EEG clip length, in seconds
        standardize: if True, will z-normalize wrt train set
        num_workers: int
        augmentation: if True, perform random augmentation on EEG
        use_fft: whether perform Fourier transform
        sampling_ratio: ratio of positive to negative examples for undersampling
        seed: random seed for undersampling
        preproc_dir: dir to preprocessed Fourier transformed data, optional
    Returns:
        dataloaders: dictionary of train/dev/test dataloaders
        datasets: dictionary of train/dev/test datasets
        scaler: standard scaler
    """
    if (graph_type is not None) and (
            graph_type not in ['individual', 'combined']):
        raise NotImplementedError

    # load mean and std
    if standardize:
        means_dir = os.path.join(
            FILEMARKER_DIR,
            'means_seq2seq_fft_' +
            str(max_seq_len) +
            's_szdetect_single.pkl')
        stds_dir = os.path.join(
            FILEMARKER_DIR,
            'stds_seq2seq_fft_' +
            str(max_seq_len) +
            's_szdetect_single.pkl')
        with open(means_dir, 'rb') as f:
            means = pickle.load(f)
        with open(stds_dir, 'rb') as f:
            stds = pickle.load(f)

        scaler = StandardScaler(mean=means, std=stds)
    else:
        scaler = None

    dataloaders = {}
    datasets = {}
    for split in ['train', 'dev', 'test']:
        if split == 'train':
            data_augment = augmentation
        else:
            data_augment = False  # never do augmentation on dev/test sets

        dataset = SeizureDataset(input_dir=input_dir,
                                 raw_data_dir=raw_data_dir,
                                 max_seq_len=max_seq_len,
                                 standardize=standardize,
                                 scaler=scaler,
                                 split=split,
                                 data_augment=data_augment,
                                 sampling_ratio=sampling_ratio,
                                 seed=seed,
                                 use_fft=use_fft,
                                 preproc_dir=preproc_dir)

        if split == 'train':
            shuffle = True
            batch_size = train_batch_size
        else:
            shuffle = False
            batch_size = test_batch_size

        loader = DataLoader(dataset=dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)
        dataloaders[split] = loader
        datasets[split] = dataset

    return dataloaders, datasets, scaler
