import os
import json
import pandas as pd
import numpy as np
import mne
import pywt
import math
import matplotlib.pyplot as plt
import time
import bisect
from scipy.signal import iirnotch, filtfilt
from constants import *

#IMPORTANT: run from root folder of repo so that cwd is AlzheimersDetection/
label_path = "data/participants.tsv"

def get_subject_group(subject_str):
    participant_id = "sub-" + subject_str
    participants_df = pd.read_csv(label_path, sep="\t")
    participant_row = participants_df[participants_df['participant_id'] == participant_id]
    assert(not participant_row.empty)
    return participant_row['Group'].values[0]

def process_epoch(global_epoch_id, is_healthy, epoch, n_segments, segment_length_samples, sampling_freq):
    sequence = []
    for segment_id in range(n_segments):
        start_sample = segment_id * STEP_LENGTH_SAMPLES
        end_sample = start_sample + segment_length_samples #exclusive
        segment = epoch[:, start_sample:end_sample]
        assert segment.shape == (N_CHANNELS, segment_length_samples)

        #notch filter
        b_notch, a_notch = iirnotch(w0=CFC_NOTCH_FREQUENCY / (sampling_freq / 2), Q=CFC_NOTCH_QUALITY_FACTOR)
        for i in range(N_CHANNELS):
            segment[i, :] = filtfilt(b_notch, a_notch, segment[i, :])
        
        sequence.append(segment)

    sequence = np.array(sequence)
    assert(sequence.shape == (n_segments, N_CHANNELS, segment_length_samples))

    #store epoch features somewhere for later use
    epoch_path = f"data/eeg_overlap/eeg_{global_epoch_id}_{'cn' if is_healthy else 'ad'}.npy"
    np.save(epoch_path, sequence)
    print(f"saved {epoch_path}")


def main():
    start_time = time.time()
    global_epoch_id = 0

    #NEMAR Dataset
    for subject_id in range(1, N_SUBJECTS + 1):
        subject_str = f"{subject_id:03}"

        #skip the FTD subjects
        subject_group = get_subject_group(subject_str) #A alzheimer, C healthy, F dontcare
        if subject_group == "F":
            continue

        subject_path = f"../data/ds004504/sub-{subject_str}/eeg/sub-{subject_str}_task-eyesclosed_eeg.set"
        json_path = f"../data/ds004504/sub-{subject_str}/eeg/sub-{subject_str}_task-eyesclosed_eeg.json"
        
        json_file =  open(json_path, 'r');
        json_data = json.load(json_file)
        recording_duration = json_data.get("RecordingDuration")
        subject_raw = mne.io.read_raw_eeglab(subject_path, preload=True)

        subject_data = subject_raw.get_data()
        assert(subject_data.shape == (N_CHANNELS, int(SAMPLING_FREQUENCY * recording_duration)))

        d_epoch_start_time_sec = (recording_duration - NEMAR_START_PADDING_LENGTH_SECONDS - EPOCH_LENGTH_SECONDS) / (N_EPOCHS - 1)
        d_epoch_start_time_sample = int(d_epoch_start_time_sec * SAMPLING_FREQUENCY)

        for epoch_id in range(N_EPOCHS):
            start_sample = NEMAR_START_PADDING_LENGTH_SAMPLES + epoch_id * d_epoch_start_time_sample
            epoch = subject_data[:, start_sample : (start_sample + EPOCH_LENGTH_SAMPLES)]
            assert(epoch.shape == (N_CHANNELS, EPOCH_LENGTH_SAMPLES))

            process_epoch(global_epoch_id, subject_group == "C", epoch, N_SEGMENTS, SEGMENT_LENGTH_SAMPLES, SAMPLING_FREQUENCY)
            global_epoch_id += 1
    
    end_time = time.time()
    print(f"time spent: {end_time - start_time}")


if __name__ == "__main__":
    main()