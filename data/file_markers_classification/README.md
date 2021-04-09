This folder contains file markers for train/validation/test sets for **seizure type classification**, and pre-computed mean and standard deviation of the training data.

In each of the files `trainSet_seizure_files.txt`, `devSet_seizure_files.txt` (i.e. validation set), and `testSet_seizure_files.txt`, each line represents `<edf-file-name>, <seizure-class>, <seizure-index>`.

For seizure class, `0` corresponds to `combined focal seizures`, `1` corresponds to `generalized non-specific seizures`, `2` corresponds to `absence seizures`, and `3` corresponds to `combined tonic seizures`.

`<seizure-index>` indicates the index of the seizure in the edf file.