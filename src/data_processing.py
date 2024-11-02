import os
import pandas as pd
from mne.io import read_epochs_eeglab

#IMPORTANT: run from root folder of repo so that cwd is AlzheimersDetection/

DATA_PATH = "data/eeg_files_notch60_50_split5_12"
LABEL_PATH = "data/participants.tsv"
SAMPLING_FREQUENCY_HZ = 500
EPOCH_LENGTH_SECONDS = 12
EPOCH_LENGTH_SAMPLES = EPOCH_LENGTH_SECONDS * SAMPLING_FREQUENCY_HZ
N_CHANNELS = 19
N_SUBJECTS = 88
N_AD = 36
N_FTD = 23
N_CN = 29
N_EPOCHS = 5
N_SEGMENTS = 6
SEGMENT_LENGTH_SAMPLES = int(EPOCH_LENGTH_SAMPLES / N_SEGMENTS)


def get_subject_group(subject_str):
    participant_id = "sub-" + subject_str
    participants_df = pd.read_csv(LABEL_PATH, sep="\t")
    participant_row = participants_df[participants_df['participant_id'] == participant_id]
    assert(not participant_row.empty)
    return participant_row['Group'].values[0]

def main():
    ad_cn_count = 0
    segment_count = 0

    for subject_id in range(1, N_SUBJECTS + 1):
        subject_str = f"{subject_id:03}"

        #skip the FTD subjects
        if get_subject_group(subject_str) == "F":
            continue

        ad_cn_count += 1
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
                segment_count += 1
    
    assert(ad_cn_count == N_AD + N_CN)
    assert(segment_count == ad_cn_count * N_EPOCHS * N_SEGMENTS)

if __name__ == "__main__":
    main()