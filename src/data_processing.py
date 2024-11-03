import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from mne.io import read_epochs_eeglab
from scipy.signal import iirnotch, filtfilt, butter

# Paths and constants
DATA_PATH = "/content/drive/MyDrive/eeg_files_notch60_50_split5_12"  # Adjust this to your Google Drive path
LABEL_PATH = "/content/drive/MyDrive/participants.tsv"
SAMPLING_FREQUENCY_HZ = 500
SAMPLING_PERIOD_SECONDS = 1 / SAMPLING_FREQUENCY_HZ
EPOCH_LENGTH_SECONDS = 12
EPOCH_LENGTH_SAMPLES = int(EPOCH_LENGTH_SECONDS * SAMPLING_FREQUENCY_HZ)
N_CHANNELS = 19
N_SUBJECTS = 88
N_EPOCHS = 5
N_SEGMENTS = 6
SEGMENT_LENGTH_SAMPLES = int(EPOCH_LENGTH_SAMPLES / N_SEGMENTS)
ALPHA_FREQUENCIES = (4, 8, 0.1)
GAMMA_FREQUENCIES = (30, 120, 1)
CWT_B = 6
CWT_C = 0.8125

# Function to calculate CWT scales based on frequencies
def get_cwt_scales(min_freq, max_freq, step_size):
    frequencies = np.arange(min_freq, max_freq, step_size)
    scales = 1 / (frequencies * SAMPLING_PERIOD_SECONDS)
    return scales

# Function to read participant group from label file
def get_subject_group(subject_str):
    participant_id = "sub-" + subject_str
    participants_df = pd.read_csv(LABEL_PATH, sep="\t")
    participant_row = participants_df[participants_df['participant_id'] == participant_id]
    assert not participant_row.empty
    return participant_row['Group'].values[0]

# Notch filter function
def apply_notch_filter(signal, fs, freq=60, quality_factor=30):
    b, a = iirnotch(freq, quality_factor, fs)
    return filtfilt(b, a, signal)
  
# def bandpass_filter(signal, lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype="band")
#     return filtfilt(b, a, signal)

# Apply CWT and extract phase and amplitude
def apply_cwt(signal, scales, wavelet_name='cmor'):
    coeffs, frequencies = pywt.cwt(signal, scales, wavelet_name, sampling_period=SAMPLING_PERIOD_SECONDS)
    return coeffs, frequencies

def extract_phase_amplitude(coeffs):
    phase = np.angle(coeffs)  # Extract phase
    amplitude = np.abs(coeffs)  # Extract amplitude
    return phase, amplitude

# PAC calculation helper functions
def calculate_pac(theta_phase, gamma_amplitude, nbins=18):
    # Bin the theta phases
    phase_bins = np.linspace(-np.pi, np.pi, nbins + 1)
    # Digitize the theta_phase array to determine which bin each phase falls into
    bin_indices = np.digitize(theta_phase, phase_bins) - 1  # `-1` to match zero-based indexing
    amplitude_means = np.zeros(nbins)

    # Calculate the mean gamma amplitude for each bin
    for i in range(nbins):
        # Select gamma_amplitude values in the current bin and calculate the mean
        bin_amplitudes = gamma_amplitude[bin_indices == i]
        amplitude_means[i] = np.mean(bin_amplitudes) if bin_amplitudes.size > 0 else 0

    # Normalize to create a probability distribution
    amplitude_prob_dist = amplitude_means / np.sum(amplitude_means)
    return amplitude_prob_dist

def calculate_modulation_index(pac_distribution):
    H = -np.sum(pac_distribution * np.log(pac_distribution + 1e-8))
    H_max = np.log(len(pac_distribution))
    modulation_index = (H_max - H) / H_max
    return modulation_index

# Main processing function for gPAC matrix generation and plotting
def generate_gpac_plots():
    alpha_scales = get_cwt_scales(ALPHA_FREQUENCIES[0], ALPHA_FREQUENCIES[1], ALPHA_FREQUENCIES[2])
    gamma_scales = get_cwt_scales(GAMMA_FREQUENCIES[0], GAMMA_FREQUENCIES[1], GAMMA_FREQUENCIES[2])
    cwt_scales = np.concatenate((alpha_scales, gamma_scales))
    
    for subject_id in range(1, N_SUBJECTS + 1):
        subject_str = f"{subject_id:03}"
        if get_subject_group(subject_str) == "F":  # Skip FTD group
            continue

        dataset_path = os.path.join(DATA_PATH, f"split-{subject_str}.set")
        epochs = read_epochs_eeglab(dataset_path)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create 2x3 grid for six segments
        for epoch_id in range(N_EPOCHS):
            epoch = epochs[epoch_id].get_data()[0]
            for segment_id in range(N_SEGMENTS):
                start_sample = segment_id * SEGMENT_LENGTH_SAMPLES
                end_sample = start_sample + SEGMENT_LENGTH_SAMPLES
                segment = epoch[:, start_sample:end_sample]

                filtered_segment = np.zeros_like(segment)
                for channel_id in range(N_CHANNELS):
                    notch_filtered_signal = apply_notch_filter(segment[channel_id, :], SAMPLING_FREQUENCY_HZ)

                # Placeholder PAC matrix
                pac_matrix = np.zeros((len(alpha_scales), len(gamma_scales)))

                # Calculate PAC for each channel and aggregate to gPAC
                for channel_id in range(N_CHANNELS):
                    segment_channel = segment[channel_id, :]
                    coefs, _ = apply_cwt(segment_channel, cwt_scales, f"cmor{CWT_B}-{CWT_C}")
                    
                    alpha_coefs = coefs[:len(alpha_scales), :]
                    gamma_coefs = coefs[len(alpha_scales):, :]

                    alpha_phase, _ = extract_phase_amplitude(alpha_coefs)
                    _, gamma_amplitude = extract_phase_amplitude(gamma_coefs)

                    min_time_points = min(alpha_phase.shape[1], gamma_amplitude.shape[1])
                    alpha_phase = alpha_phase[:, :min_time_points]
                    gamma_amplitude = gamma_amplitude[:, :min_time_points]

                    for a_idx, theta_phase in enumerate(alpha_phase):
                        for g_idx, gamma_amp in enumerate(gamma_amplitude):
                            pac_dist = calculate_pac(theta_phase, gamma_amp)
                            pac_matrix[a_idx, g_idx] += calculate_modulation_index(pac_dist)

                pac_matrix /= N_CHANNELS  # Average across channels

                # Plot PAC matrix for the segment
                ax = axes[segment_id // 3, segment_id % 3]
                cax = ax.imshow(pac_matrix.T, extent=[ALPHA_FREQUENCIES[0], ALPHA_FREQUENCIES[1], GAMMA_FREQUENCIES[0], GAMMA_FREQUENCIES[1]], aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f"Segment {segment_id}")
                ax.set_xlabel("Theta Phase Frequency (Hz)")
                ax.set_ylabel("Gamma Amplitude Frequency (Hz)")
                fig.colorbar(cax, ax=ax, label="PAC Modulation Index")

            break  # Process one epoch per subject for demonstration
        plt.tight_layout()
        plt.show()
        break  # Process one subject for demonstration

# Run the main function
generate_gpac_plots()
