import os
from mne import Epochs, find_events
from mne.io import read_epochs_eeglab


DATA_PATH = "../eeg_files_notch60_50_split5_12"
SAMPLING_FREQUENCY_HZ = 500
EPOCH_LENGTH_SECONDS = 12
EPOCH_LENGTH_SAMPLES = EPOCH_LENGTH_SECONDS * SAMPLING_FREQUENCY_HZ
N_CHANNELS = 19
N_SUBJECTS = 88
N_EPOCHS = 5
N_SEGMENTS = 6
SEGMENT_LENGTH_SAMPLES = int(EPOCH_LENGTH_SAMPLES / N_SEGMENTS)


for subject_id in range(1, N_SUBJECTS + 1):
    subject_str = f"{subject_id:03}"
    dataset_path = os.path.join(DATA_PATH, f"split-{subject_str}.set")

    epochs = read_epochs_eeglab(dataset_path)

    for epoch_id in range(N_EPOCHS):
        epoch = epochs[epoch_id].get_data()[0] #a numpy array
        assert(epoch.shape == (N_CHANNELS, EPOCH_LENGTH_SAMPLES))

        for segment_id in range(N_SEGMENTS):
            start_sample = segment_id * SEGMENT_LENGTH_SAMPLES
            end_sample = start_sample + SEGMENT_LENGTH_SAMPLES #exclusive
            segment = epoch[:, start_sample:end_sample]

            #TODO: do something with the segment
            print(segment.shape)
