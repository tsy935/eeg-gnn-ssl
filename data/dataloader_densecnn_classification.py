"""Dataloader for Dense-CNN"""
import sys
sys.path.append('../')
from pathlib import Path
import scipy.signal
import scipy
import pickle
import os
import numpy as np
import h5py
import math
import torch
from torch.utils.data import Dataset, DataLoader
from utils import StandardScaler
from constants import INCLUDED_CHANNELS, FREQUENCY
from data.data_utils import *
import utils
import pyedflib

repo_paths = str(Path.cwd()).split('eeg-gnn-ssl')
repo_paths = Path(repo_paths[0]).joinpath('eeg-gnn-ssl')
sys.path.append(repo_paths)
FILEMARKER_DIR = Path(repo_paths).joinpath('data/file_markers_classification')


def computeSliceMatrix(
        h5_fn,
        edf_fn,
        seizure_idx,
        clip_len=60,
        is_fft=False):
    """
    Comvert entire EEG sequence into clips of length clip_len
    Args:
        h5_fn: file name of resampled signal h5 file (full path)
        edf_fn: full path to edf file
        seizure_idx: current seizure index in edf file, int
        clip_len: sliding window size or EEG clip length, in seconds, int
        is_fft: whether to perform FFT on raw EEG data
    Returns:
        eeg_clip: eeg clip (clip_len, num_channels, time_step_size*freq)
    """
    offset = 2 # hard-coded offset

    with h5py.File(h5_fn, 'r') as f:
        signal_array = f["resampled_signal"][()] # (num_channels, num_data_points)
        resampled_freq = f["resample_freq"][()]
    assert resampled_freq == FREQUENCY

    # get seizure times
    seizure_times = getSeizureTimes(edf_fn.split('.edf')[0])
    curr_seizure_time = seizure_times[seizure_idx]

    if seizure_idx > 0:
        pre_seizure_end = int(FREQUENCY * seizure_times[seizure_idx - 1][1])
    else:
        pre_seizure_end = 0

    # start_time: start of current seizure - offset / end of previous seizure, whichever comes later
    start_t = max(pre_seizure_end + 1, int(FREQUENCY*(curr_seizure_time[0] - offset)))
    # end_time: (start_time + clip_len) / end of current seizure, whichever comes first    
    end_t = min(start_t + int(FREQUENCY*clip_len), int(FREQUENCY*curr_seizure_time[1]))

    # get corresponding eeg clip
    signal_array = signal_array[:, start_t:end_t]

    if is_fft:
        eeg_clip, _ = computeFFT(signal_array, n=signal_array.shape[-1])

    ## Padding zeros
    seq_len = eeg_clip.shape[-1]
    if is_fft:
        diff = int(FREQUENCY * clip_len/2) - eeg_clip.shape[-1]
    else:
        diff = FREQUENCY * clip_len - eeg_clip.shape[-1]    
    if diff > 0:
        zeros = np.zeros((eeg_clip.shape[0], diff))
        eeg_clip = np.concatenate((eeg_clip, zeros), axis=1)
    elif diff < 0:
        eeg_clip = eeg_clip[:,:int(FREQUENCY * window_size)]

    return eeg_clip.T, seq_len


class SeizureDataset(Dataset):
    def __init__(
            self,
            input_dir,
            raw_data_dir,
            max_seq_len=60,
            standardize=True,
            scaler=None,
            split='train',
            padding_val=0,
            data_augment=False,
            use_fft=False,
            preproc_dir=None):
        """
        Args:
            input_dir: dir to resampled signals h5 files
            raw_data_dir: dir to TUSZ edf files
            time_step_size: int, in seconds
            max_seq_len: int, EEG clip length, in seconds
            standardize: if True, will z-normalize wrt train set
            scaler: scaler object for standardization
            split: train, dev or test
            padding_val: int, value used for padding to max_seq_len
            data_augment: if True, perform random augmentation of EEG
            use_fft: whether perform Fourier transform
        """
        if standardize and (scaler is None):
            raise ValueError('To standardize, please provide scaler.')

        self.input_dir = input_dir
        self.raw_data_dir = raw_data_dir
        self.max_seq_len = max_seq_len
        self.standardize = standardize
        self.scaler = scaler
        self.split = split
        self.padding_val = padding_val
        self.data_augment = data_augment
        self.use_fft = use_fft
        self.preproc_dir = preproc_dir

        # get full paths to all raw edf files
        self.edf_files = []
        for path, subdirs, files in os.walk(raw_data_dir):
            for name in files:
                if ".edf" in name:
                    self.edf_files.append(os.path.join(path, name))

        # read file tuples: (edf_fn, seizure_class, seizure_idx)
        file_marker_dir = os.path.join(FILEMARKER_DIR, split+"Set_seizure_files.txt")
        with open(file_marker_dir, 'r') as f:
            f_str = f.readlines()
        
        self.file_tuples = []
        for i in range(len(f_str)):
            tup = f_str[i].strip("\n").split(",")
            tup[1] = int(tup[1]) # seizure class
            tup[2] = int(tup[2]) # seizure index
            self.file_tuples.append(tup)
        self.size = len(self.file_tuples)

        # get sensor ids
        self.sensor_ids = [x.split(' ')[-1] for x in INCLUDED_CHANNELS]

    def __len__(self):
        return self.size

    def _random_reflect(self, eeg_clip):
        swap_pairs = get_swap_pairs(INCLUDED_CHANNELS)
        eeg_clip_reflect = eeg_clip.copy()
        if(np.random.choice([True, False])):            
            for pair in swap_pairs:
                eeg_clip_reflect[:,[pair[0],pair[1]]] = eeg_clip[:,[pair[1], pair[0]]]

        return eeg_clip_reflect
    
    def _random_scale(self, eeg_clip):
        scale_factor = np.random.uniform(0.8, 1.2)
        if self.use_fft:
            eeg_clip += np.log(scale_factor)
        else:
            eeg_clip *= scale_factor
        return eeg_clip


    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (x, y, seq_len, supports, adj_mat, write_file_name)
        """
        edf_fn, seizure_class, seizure_idx = self.file_tuples[idx]
        seizure_idx = int(seizure_idx)

        # find edf file full path
        edf_file = [file for file in self.edf_files if edf_fn in file]
        assert len(edf_file) == 1
        edf_file = edf_file[0]

        # preprocess
        if self.preproc_dir is not None: # load preprocessed
            h5_fn = edf_fn.split('/')[-1] + '_' + str(seizure_idx) + '.h5'
            with h5py.File(os.path.join(self.preproc_dir, h5_fn), 'r') as hf:
                try:
                    curr_feature = hf['clip'][()] # (seq_len, num_nodes, input_dim)
                except:
                    curr_feature = hf['features'][()] # (seq_len, num_nodes, input_dim)
            # padding
            curr_len = curr_feature.shape[0]
            seq_len = np.minimum(curr_len, self.max_seq_len)
            if curr_len < self.max_seq_len:
                len_pad = self.max_seq_len - curr_len
                padded_feature = np.ones((len_pad, curr_feature.shape[1], curr_feature.shape[2])) * 0.
                eeg_clip = np.concatenate((curr_feature, padded_feature), axis=0)
            else:
                eeg_clip = curr_feature[:self.max_seq_len,:]
            eeg_clip = np.transpose(eeg_clip, (1, 0, 2)).reshape(19, -1).T
        else:
            resample_sig_dir = os.path.join(
                self.input_dir, edf_fn.split('.edf')[0] + '.h5')
            eeg_clip, seq_len = computeSliceMatrix(
                h5_fn=resample_sig_dir, edf_fn=edf_file, seizure_idx=seizure_idx,
                clip_len=self.max_seq_len, is_fft=self.use_fft
            ) # already padded
            
        # data augmentation
        if self.data_augment:
            eeg_clip = self._random_reflect(eeg_clip)
            eeg_clip = self._random_scale(eeg_clip)

        # standardize wrt train mean and std
        if self.standardize:
            eeg_clip = self.scaler.transform(eeg_clip)

        # convert to tensors
        x = torch.FloatTensor(eeg_clip)
        y = torch.LongTensor([seizure_class])
        seq_len = torch.LongTensor([seq_len])
        writeout_fn = edf_fn + "_" + str(seizure_idx)

        return (x, y, seq_len, [], [], writeout_fn) # for dataloader consistency


def load_dataset_densecnn_classification(
        input_dir,
        raw_data_dir,
        train_batch_size,
        test_batch_size=None,
        max_seq_len=60,
        standardize=True,
        num_workers=8,
        padding_val=0.,
        augmentation=False,
        use_fft=False,
        preproc_dir=None):
    """
    Args:
        input_dir: dir to resampled signals h5 files
        raw_data_dir: dir to TUSZ raw edf files
        train_batch_size: int
        test_batch_size: int
        max_seq_len: EEG clip length, in seconds
        standardize: if True, will z-normalize wrt train set
        num_workers: int
        padding_val: value used for padding
        augmentation: if True, perform random augmentation of EEG
        use_fft: whether perform Fourier transform
        preproc_dir: dir to preprocessed Fourier transformed data, optional
    Returns:
        dataloaders: dictionary of train/dev/test dataloaders
        datasets: dictionary of train/dev/test datasets
        scaler: standard scaler
    """

    # load per-node mean and std
    if standardize:
        means_dir = os.path.join(
            FILEMARKER_DIR, 'means_fft_'+str(max_seq_len)+'s_single.pkl')
        stds_dir = os.path.join(
            FILEMARKER_DIR, 'stds_fft_'+str(max_seq_len)+'s_single.pkl')
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
            data_augment = False  # no augmentation on dev/test sets

        dataset = SeizureDataset(input_dir=input_dir,
                                 raw_data_dir=raw_data_dir,
                                 max_seq_len=max_seq_len,
                                 standardize=standardize,
                                 scaler=scaler,
                                 split=split,
                                 padding_val=padding_val,
                                 data_augment=data_augment,
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
