import os
import sys
import argparse
import h5py
from tqdm import tqdm
from dataloader_classification import computeSliceMatrix

FILE_MARKER_DIR = "./file_markers_classification"

def main(
        resample_dir,
        raw_data_dir,
        output_dir,
        clip_len,
        time_step_size,
        is_fft=True):
    with open(os.path.join(
            FILE_MARKER_DIR, "trainSet_seizure_files.txt"), "r") as f:
        train_str = f.readlines()
    train_tuples = [curr_str.strip('\n').split(',') for curr_str in train_str]

    with open(os.path.join(
            FILE_MARKER_DIR, "devSet_seizure_files.txt"), "r") as f:
        dev_str = f.readlines()
    dev_tuples = [curr_str.strip('\n').split(',') for curr_str in dev_str]

    with open(os.path.join(
            FILE_MARKER_DIR, "testSet_seizure_files.txt"), "r") as f:
        test_str = f.readlines()
    test_tuples = [curr_str.strip('\n').split(',') for curr_str in test_str]

    all_tuples = train_tuples + dev_tuples + test_tuples

    edf_files = []
    for path, subdirs, files in os.walk(raw_data_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    output_dir = os.path.join(
        output_dir,
        'clipLen' +
        str(clip_len) +
        '_timeStepSize' +
        str(time_step_size))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    delayed_results = []
    delayed_files = []
    for idx in tqdm(range(len(all_tuples))):
        edf_fn, _, seizure_idx = all_tuples[idx]
        seizure_idx = int(seizure_idx)
        edf_fn_full = [file for file in edf_files if edf_fn in file][0]
        h5_fn = edf_fn.split('.edf')[0] + '.h5'

        eeg_clip = computeSliceMatrix(
            h5_fn=os.path.join(resample_dir, h5_fn),
            edf_fn=edf_fn_full,
            seizure_idx=seizure_idx,
            time_step_size=time_step_size,
            clip_len=clip_len,
            is_fft=is_fft)

        with h5py.File(os.path.join(output_dir, edf_fn + '_' + str(seizure_idx) + '.h5'), 'w') as hf:
            hf.create_dataset('clip', data=eeg_clip)

    print("Preprocessing DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resampled_dir",
        type=str,
        default=None,
        help="Directory to resampled signals.")
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Directory to raw edf files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory.")
    parser.add_argument(
        "--clip_len",
        type=int,
        default=60,
        help="EEG clip length in seconds.")
    parser.add_argument(
        "--time_step_size",
        type=int,
        default=1,
        help="Time step size in seconds.")
    parser.add_argument(
        "--is_fft",
        action="store_true",
        default=False,
        help="Whether to perform FFT.")

    args = parser.parse_args()
    main(
        args.resampled_dir,
        args.raw_data_dir,
        args.output_dir,
        args.clip_len,
        args.time_step_size,
        args.is_fft)
