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
from scipy.signal import butter, filtfilt
from constants import *

#IMPORTANT: run from root folder of repo so that cwd is AlzheimersDetection/
label_path = "data/participants.tsv"

def get_subject_group(subject_str):
    participant_id = "sub-" + subject_str
    participants_df = pd.read_csv(label_path, sep="\t")
    participant_row = participants_df[participants_df['participant_id'] == participant_id]
    assert(not participant_row.empty)
    return participant_row['Group'].values[0]

def get_band_index(frequency):
    insertion_id = bisect.bisect_left(FREQUENCY_BANDS, frequency)
    return insertion_id - 1 if insertion_id > 0 else 0

def plot_segment_fft(segment_fft_features):
    plt.figure(figsize=(10, 8))  # Set the figure size
    plt.imshow(segment_fft_features, cmap="coolwarm", aspect="auto")  # Display the array as an image

    # Add a color bar
    plt.colorbar(label="Value")

    # Add labels and title
    plt.title("Heatmap of 19x5 Array")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

def process_epoch(global_epoch_id, is_healthy, epoch, n_segments, segment_length_samples, sampling_period_seconds):
    if global_epoch_id <= 200:
        return
    epoch_fft_features = []
    
    for segment_id in range(n_segments):
        start_sample = segment_id * segment_length_samples
        end_sample = start_sample + segment_length_samples #exclusive
        segment = epoch[:, start_sample:end_sample]
        assert segment.shape == (N_CHANNELS, segment_length_samples)
        segment_fft_features = []

        for channel_id in range(N_CHANNELS):
            segment_channel = segment[channel_id, :]
            assert segment_channel.shape == (segment_length_samples,)

            #LP filter
            nyquist = 1 / (2 * sampling_period_seconds)
            normal_cutoff = FFT_LP_FREQUENCY / nyquist
            b_lp, a_lp = butter(FFT_LP_ORDER, normal_cutoff, btype='low')
            segment_channel_lp = filtfilt(b_lp, a_lp, segment_channel)

            #find FFT coefficients
            fft_coefs = np.fft.fft(segment_channel_lp)
            all_freqs = np.fft.fftfreq(segment_length_samples, d=sampling_period_seconds)
            mask = (all_freqs >= FREQUENCY_BANDS[0]) & (all_freqs <= FREQUENCY_BANDS[-1])
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
        epoch_fft_features.append(segment_fft_features)
        plot_segment_fft(segment_fft_features)
        
    epoch_fft_features = np.array(epoch_fft_features)
    assert(epoch_fft_features.shape == (n_segments, N_CHANNELS, N_FREQUENCY_BANDS))

    #store epoch features somewhere for later use
    #epoch_path = f"data/fft/fft_{global_epoch_id}_{'cn' if is_healthy else 'ad'}.npy"
    #np.save(epoch_path, epoch_fft_features)
    #print(f"saved {epoch_path}")


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

        #inspect raw signals
        #subject_raw.plot(scalings="auto", show=True, block=True)
        #channel_names = subject_raw.info['ch_names']
        #print("Channel Names:", channel_names)

        #filtering; NAH using scipy now
        #subject_raw.filter(l_freq=None, h_freq=50)

        subject_data = subject_raw.get_data()
        assert(subject_data.shape == (N_CHANNELS, int(SAMPLING_FREQUENCY * recording_duration)))

        d_epoch_start_time_sec = (recording_duration - 2 * NEMAR_PADDING_LENGTH_SECONDS - EPOCH_LENGTH_SECONDS) / (N_EPOCHS - 1)
        d_epoch_start_time_sample = int(d_epoch_start_time_sec * SAMPLING_FREQUENCY)

        for epoch_id in range(N_EPOCHS):
            start_sample = NEMAR_PADDING_LENGTH_SAMPLES + epoch_id * d_epoch_start_time_sample
            epoch = subject_data[:, start_sample : (start_sample + EPOCH_LENGTH_SAMPLES)]
            assert(epoch.shape == (N_CHANNELS, EPOCH_LENGTH_SAMPLES))

            process_epoch(global_epoch_id, subject_group == "C", epoch, N_SEGMENTS, SEGMENT_LENGTH_SAMPLES, SAMPLING_PERIOD_SECONDS)
            global_epoch_id += 1

    #OSF Dataset
    for alz_id in range(OSF_N_AD):
        subject_path = f"../data/EEG_data/AD/Eyes_closed/Paciente{alz_id+1}/"
        epoch = []

        for channel_name in CHANNEL_NAMES:
            channel_path = subject_path + channel_name + ".txt"
            channel = np.loadtxt(channel_path)
            assert(channel.shape == (OSF_SAMPLE_LENGTH_SAMPLES,))
            epoch.append(channel)
            #plot_signal(channel)
        
        epoch = np.array(epoch)
        process_epoch(global_epoch_id, False, epoch, OSF_N_SEGMENTS, OSF_SEGMENT_LENGTH_SAMPLES, OSF_SAMPLING_PERIOD_SECONDS)
        global_epoch_id += 1

    for healthy_id in range(OSF_N_CN):
        subject_path = f"../data/EEG_data/Healthy/Eyes_closed/Paciente{healthy_id+1}/"
        epoch = []

        for channel_name in CHANNEL_NAMES:
            channel_path = subject_path + channel_name + ".txt"
            channel = np.loadtxt(channel_path)
            assert(channel.shape == (OSF_SAMPLE_LENGTH_SAMPLES,))
            epoch.append(channel)
        
        epoch = np.array(epoch)
        process_epoch(global_epoch_id, True, epoch, OSF_N_SEGMENTS, OSF_SEGMENT_LENGTH_SAMPLES, OSF_SAMPLING_PERIOD_SECONDS)
        global_epoch_id += 1
    
    end_time = time.time()
    print(f"time spent: {end_time - start_time}")


if __name__ == "__main__":
    main()