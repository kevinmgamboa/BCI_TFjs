# BCI_TFjs

**NOTE:** This repository contains code for the Alex Project. For more info visit [Brains@Play Discord](https://brainsatplay.com/).

### Training a model
For training a new model, run the `train.py` file. To develop a new model check the format in `modelhub.py`. You can also create directly in the `train.py` file by replacing the class that calls the model with your algorithm.

### Alex Dataset

The EEG files in `.edf` format (or Alex dataset) must be renamed adding **_sleep** or **_awake** at the end of the file. For example `eegfile.edf` -> `eegfile_sleep.edf` when the EEG signal was recorded when sleeping. The files can be saved in:

    /datasets/alex/
        eegfile1_sleep.edf
        eegfile2_sleep.edf
        eegfile3_awake.edf
        eegfile4_awake.edf
