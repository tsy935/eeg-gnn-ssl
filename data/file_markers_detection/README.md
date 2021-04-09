This folder contains file markers for train/validation/test sets for **seizure detection**, and pre-computed mean and standard deviation of the training data.

In each of the text files, each line represents `<edf-file-name>_<clip-index>.h5, <binary-seizure-label>`, where `<clip-index>` indicates the index of the EEG clip in the edf file, `<binary-seizure-label>` is 1 if there is at least one seizure within this clip, otherwise 0.