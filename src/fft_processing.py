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
from constants import *

#IMPORTANT: run from root folder of repo so that cwd is AlzheimersDetection/
label_path = "data/participants.tsv"

def get_subject_group(subject_str):
    participant_id = "sub-" + subject_str
    participants_df = pd.read_csv(label_path, sep="\t")
    participant_row = participants_df[participants_df['participant_id'] == participant_id]
    assert(not participant_row.empty)
    return participant_row['Group'].values[0]

def get_cwt_scales(min_freq, max_freq, step_size):
    frequencies = np.arange(min_freq, max_freq, step_size)
    scales = 1 / (frequencies * SAMPLING_PERIOD_SECONDS)
    return scales

def get_phase_bin(phase_rads):
    phase_rads = np.where(phase_rads == math.pi, -math.pi, phase_rads)
    assert np.all((phase_rads >= -math.pi) & (phase_rads < math.pi))
    bin_indices = ((phase_rads + math.pi) / (2 * math.pi) * N_PHASE_BINS).astype(int)
    assert np.all((bin_indices >= 0) & (bin_indices < N_PHASE_BINS))
    return bin_indices

def get_band_index(frequency):
    insertion_id = bisect.bisect_left(FREQUENCY_BANDS, frequency)
    return insertion_id - 1 if insertion_id > 0 else 0

def main():
    start_time = time.time()
    
    for subject_id in range(12, N_SUBJECTS + 1):
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

        #inspect raw signals
        #subject_raw.plot(scalings="auto", show=True, block=True)

        #filtering
        #subject_raw.notch_filter(freqs=[60])
        subject_raw.filter(l_freq=None, h_freq=50)

        subject_data = subject_raw.get_data()
        assert(subject_data.shape == (N_CHANNELS, int(SAMPLING_FREQUENCY_HZ * recording_duration)))

        d_epoch_start_time_sec = (recording_duration - 2 * SUBJECT_PADDING_LENGTH_SECONDS - EPOCH_LENGTH_SECONDS) / (N_EPOCHS - 1)
        d_epoch_start_time_sample = int(d_epoch_start_time_sec * SAMPLING_FREQUENCY_HZ)

        for epoch_id in range(N_EPOCHS):
            start_sample = SUBJECT_PADDING_LENGTH_SAMPLES + epoch_id * d_epoch_start_time_sample
            epoch = subject_data[:, start_sample : (start_sample + EPOCH_LENGTH_SAMPLES)]
            assert(epoch.shape == (N_CHANNELS, EPOCH_LENGTH_SAMPLES))
            
            for segment_id in range(N_SEGMENTS):
                start_sample = segment_id * SEGMENT_LENGTH_SAMPLES
                end_sample = start_sample + SEGMENT_LENGTH_SAMPLES #exclusive
                segment = epoch[:, start_sample:end_sample]
                assert segment.shape == (N_CHANNELS, SEGMENT_LENGTH_SAMPLES)
                segment_fft_features = []

                for channel_id in range(N_CHANNELS):
                    segment_channel = segment[channel_id, :]
                    assert segment_channel.shape == (SEGMENT_LENGTH_SAMPLES,)

                    #find FFT coefficients
                    fft_coefs = np.fft.fft(segment_channel)
                    all_freqs = np.fft.fftfreq(SEGMENT_LENGTH_SAMPLES, d=SAMPLING_PERIOD_SECONDS)
                    mask = (all_freqs >= 0) & (all_freqs <= 50)
                    filtered_freqs = all_freqs[mask]
                    filtered_fft = fft_coefs[mask]

                    #find FFT features
                    fft_features = np.zeros(N_FREQUENCY_BANDS)
                    for freq, coef in zip(filtered_freqs, filtered_fft):
                        band_id = get_band_index(freq)
                        fft_features[band_id] += abs(coef)
                    for band_id in range(N_FREQUENCY_BANDS):
                        fft_features[band_id] /= FREQUENCY_BANDS[band_id + 1] - FREQUENCY_BANDS[band_id]
                    segment_fft_features.append(fft_features)
                
                segment_fft_features = np.array(segment_fft_features)
                assert(segment_fft_features.shape == (N_CHANNELS, N_FREQUENCY_BANDS))

                
                break
            break
        break
    
    end_time = time.time()
    print(f"time spent: {end_time - start_time}")


if __name__ == "__main__":
    main()