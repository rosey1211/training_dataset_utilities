This module will remove all augmented files listed in the labels.csv file for a dataset.  These are files whose names contain a _c or _s or _f.  The dataset is specified as a command line: ./clean_labels -ds <dataset_name>

It will save in place and destructively remove those files.