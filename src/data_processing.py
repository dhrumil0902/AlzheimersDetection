import os
import json
import pandas as pd
import numpy as np
import mne
import pywt
import math
import matplotlib.pyplot as plt
import time
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

def main():
    start_time = time.time()

    alpha_scales = get_cwt_scales(ALPHA_FREQUENCIES[0], ALPHA_FREQUENCIES[1], ALPHA_FREQUENCIES[2])
    gamma_scales = get_cwt_scales(GAMMA_FREQUENCIES[0], GAMMA_FREQUENCIES[1], GAMMA_FREQUENCIES[2])
    cwt_scales = np.concatenate((alpha_scales, gamma_scales))
    
    for subject_id in range(12, N_SUBJECTS + 1):
        subject_str = f"{subject_id:03}"

        #skip the FTD subjects
        subject_group = get_subject_group(subject_str) #A alzheimer, C healthy, F dontcare
        if subject_group == "F":
            continue

        subject_path = f"../data/eeg_files_unfiltered/sub-{subject_str}_task-eyesclosed_eeg.set"
        json_path = f"../data/ds004504/sub-{subject_str}/eeg/sub-{subject_str}_task-eyesclosed_eeg.json"
        
        json_file =  open(json_path, 'r');
        json_data = json.load(json_file)
        recording_duration = json_data.get("RecordingDuration")
        subject_raw = mne.io.read_raw_eeglab(subject_path, preload=True)

        #notch filter
        #subject_raw.notch_filter(freqs=60)

        subject_data = subject_raw.get_data()
        assert(subject_data.shape == (N_CHANNELS, int(SAMPLING_FREQUENCY_HZ * recording_duration)))

        '''
        #plot subject data to determine how much start/end padding to use
        t = np.linspace(0, 8, 8 * SAMPLING_FREQUENCY_HZ)
        plt.figure(figsize=(10, 4))  # Optional: Set the figure size
        plt.plot(t, subject_data[0][:8*SAMPLING_FREQUENCY_HZ], label="Signal")
        plt.title(f"1D Signal {recording_duration}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()  # Show legend
        plt.grid(True)  # Add grid for better readability
        plt.show()
        continue
        '''

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
                global_pac_mi = np.zeros((N_ALPHA, N_GAMMA))

                for channel_id in range(N_CHANNELS):
                    segment_channel = segment[channel_id, :]
                    assert segment_channel.shape == (SEGMENT_LENGTH_SAMPLES,)

                    #calculate W(s, t) matrix
                    coefs, freqs = pywt.cwt(segment_channel, cwt_scales, f"cmor{CWT_B}-{CWT_C}", sampling_period=SAMPLING_PERIOD_SECONDS)
                    #coefs, freqs = pywt.cwt(segment_channel, cwt_scales, f"cmor{CWT_B}-{CWT_C}")
                    
                    #extract alpha freq phases
                    alpha_coefs = coefs[0:N_ALPHA, :]
                    #alpha_coefs, _ = pywt.cwt(data=segment_channel, scales=alpha_scales, wavelet=f"cmor{CWT_B}-{CWT_C}")
                    phases = np.angle(alpha_coefs)

                    #extract gamma freq amplitudes
                    gamma_coefs = coefs[N_ALPHA:, :]
                    #gamma_coefs, _ = pywt.cwt(data=segment_channel, scales=gamma_scales, wavelet=f"cmor{CWT_B}-{CWT_C}")
                    amplitudes = np.abs(gamma_coefs)

                    '''
                    # Create the figure
                    plt.figure(figsize=(12, 12))
                    # Plot the signal
                    t = np.linspace(0, int(SEGMENT_LENGTH_SAMPLES/SAMPLING_FREQUENCY_HZ), SEGMENT_LENGTH_SAMPLES)
                    plt.subplot(3, 1, 1)
                    plt.plot(t, segment_channel, label="Signal", color='blue')
                    plt.title("Signal")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude")
                    plt.legend()
                    plt.grid(True)
                    # Plot the alpha coefficients heatmap
                    plt.subplot(3, 1, 2)
                    plt.imshow(
                        np.abs(alpha_coefs), 
                        aspect='auto', 
                        cmap='viridis', 
                        extent=[t[0], t[-1], N_ALPHA, 0]
                    )
                    plt.colorbar(label="Magnitude")
                    plt.title("Alpha Coefficients Heatmap")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Scale")
                    # Plot the gamma coefficients heatmap
                    plt.subplot(3, 1, 3)
                    plt.imshow(
                        np.abs(gamma_coefs), 
                        aspect='auto', 
                        cmap='viridis', 
                        extent=[t[0], t[-1], len(cwt_scales) - N_ALPHA, 0]
                    )
                    plt.colorbar(label="Magnitude")
                    plt.title("Gamma Coefficients Heatmap")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Scale")
                    # Adjust layout
                    plt.tight_layout()
                    plt.show()
                    '''

                    #phase binning
                    pac_bin_mean_amplitudes = np.zeros((N_ALPHA, N_GAMMA, N_PHASE_BINS))
                    for alpha_id in range(N_ALPHA):
                        alpha_phases = phases[alpha_id, :]

                        bin_indices = get_phase_bin(alpha_phases)
                        bin_counts = np.bincount(bin_indices, minlength=N_PHASE_BINS)
                        assert np.all(bin_counts != 0)

                        for gamma_id in range(N_GAMMA):
                            gamma_amplitudes = amplitudes[gamma_id, :]
                            np.add.at(pac_bin_mean_amplitudes[alpha_id, gamma_id], bin_indices, gamma_amplitudes)
                            pac_bin_mean_amplitudes[alpha_id, gamma_id] /= bin_counts

                    #probability distribution
                    pac_distributions = pac_bin_mean_amplitudes / np.sum(pac_bin_mean_amplitudes, axis=-1, keepdims=True)
                    pac_entropies = -np.sum(pac_distributions * np.log2(pac_distributions), axis=-1)
                    pac_mi = (H_MAX - pac_entropies) / H_MAX
                    assert np.all((pac_mi >= 0) & (pac_mi <= 1))
                    global_pac_mi += pac_mi
                global_pac_mi /= N_CHANNELS
                
                # plot the global pac
                aligned_global_pac_mi = np.flip(global_pac_mi.T, axis=0)
                y_indices = np.arange(aligned_global_pac_mi.shape[0])  # First index (0 to 19)
                x_indices = np.arange(aligned_global_pac_mi.shape[1])  # Second index (0 to 39)
                y = GAMMA_FREQUENCIES[0] + GAMMA_FREQUENCIES[2] * y_indices[:, np.newaxis]  # Make y a column vector for broadcasting
                x = ALPHA_FREQUENCIES[0] + ALPHA_FREQUENCIES[2] * x_indices  # Keep x as a row vector

                plt.figure(figsize=(10, 6))
                plt.pcolormesh(x, y, aligned_global_pac_mi, shading='auto', cmap='viridis')  # Color-coded representation
                plt.colorbar(label='Value')  # Add a color bar to indicate the scale
                plt.title(f'Color Plot for subject {subject_group} {subject_str} at t = {segment_id}')  # Title of the plot
                plt.xlabel('Phase Frequencies (Hz)')  # X-axis label
                plt.ylabel('Amplitude Frequencies (Hz)')  # Y-axis label
                plt.show()  # Display the plot
                break
                
            break
        break
    
    end_time = time.time()
    print(end_time - start_time)
    #assert(ad_cn_count == N_AD + N_CN)
    #assert(segment_count == ad_cn_count * N_EPOCHS * N_SEGMENTS)

if __name__ == "__main__":
    main()