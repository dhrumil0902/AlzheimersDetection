import os
import pandas as pd
import numpy as np
from mne.io import read_epochs_eeglab
import pywt
import math
import matplotlib.pyplot as plt
import time

#IMPORTANT: run from root folder of repo so that cwd is AlzheimersDetection/

DATA_PATH = "data/eeg_files_notch60_50_split5_12"
LABEL_PATH = "data/participants.tsv"
SAMPLING_FREQUENCY_HZ = 500
SAMPLING_PERIOD_SECONDS = 1 / SAMPLING_FREQUENCY_HZ
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

#from 6.2.1 of thesis
CWT_B = 6
CWT_C = 0.8125

#from 6.2.2 and 6.4.1 of thesis
ALPHA_FREQUENCIES = (2, 8, 0.1)
GAMMA_FREQUENCIES = (30, 120, 1)
N_ALPHA = int((ALPHA_FREQUENCIES[1] - ALPHA_FREQUENCIES[0]) / ALPHA_FREQUENCIES[2])
N_GAMMA = int((GAMMA_FREQUENCIES[1] - GAMMA_FREQUENCIES[0]) / GAMMA_FREQUENCIES[2])

#from ref [26] of thesis
N_PHASE_BINS = 18
H_MAX = math.log2(N_PHASE_BINS)



def get_subject_group(subject_str):
    participant_id = "sub-" + subject_str
    participants_df = pd.read_csv(LABEL_PATH, sep="\t")
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
    #ad_cn_count = 0
    #segment_count = 0
    start_time = time.time()

    alpha_scales = get_cwt_scales(ALPHA_FREQUENCIES[0], ALPHA_FREQUENCIES[1], ALPHA_FREQUENCIES[2])
    gamma_scales = get_cwt_scales(GAMMA_FREQUENCIES[0], GAMMA_FREQUENCIES[1], GAMMA_FREQUENCIES[2])
    cwt_scales = np.concatenate((alpha_scales, gamma_scales))

    for subject_id in range(1, N_SUBJECTS + 1):
        subject_str = f"{subject_id:03}"

        #skip the FTD subjects
        subject_group = get_subject_group(subject_str)
        if subject_group == "F":
            continue

        #ad_cn_count += 1
        dataset_path = os.path.join(DATA_PATH, f"split-{subject_str}.set")
        epochs = read_epochs_eeglab(dataset_path)

        for epoch_id in range(N_EPOCHS):
            epoch = epochs[epoch_id].get_data()[0] #a numpy array
            assert(epoch.shape == (N_CHANNELS, EPOCH_LENGTH_SAMPLES))

            for segment_id in range(N_SEGMENTS):
                start_sample = segment_id * SEGMENT_LENGTH_SAMPLES
                end_sample = start_sample + SEGMENT_LENGTH_SAMPLES #exclusive
                segment = epoch[:, start_sample:end_sample]
                assert segment.shape == (N_CHANNELS, SEGMENT_LENGTH_SAMPLES)
                #segment_count += 1
                global_pac_mi = np.zeros((N_ALPHA, N_GAMMA))

                for channel_id in range(N_CHANNELS):
                    segment_channel = segment[channel_id, :]
                    assert segment_channel.shape == (SEGMENT_LENGTH_SAMPLES,)

                    #calculate W(s, t) matrix
                    coefs, freqs = pywt.cwt(segment_channel, cwt_scales, f"cmor{CWT_B}-{CWT_C}", sampling_period=SAMPLING_PERIOD_SECONDS)
                    
                    #extract alpha freq phases
                    alpha_coefs = coefs[0:N_ALPHA, :]
                    phases = np.angle(alpha_coefs)

                    #extract gamma freq amplitudes
                    gamma_coefs = coefs[N_ALPHA:, :]
                    amplitudes = np.abs(gamma_coefs)

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
                '''
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
                #break
                '''
            #break
        break
    
    end_time = time.time()
    print(end_time - start_time)
    #assert(ad_cn_count == N_AD + N_CN)
    #assert(segment_count == ad_cn_count * N_EPOCHS * N_SEGMENTS)

if __name__ == "__main__":
    main()