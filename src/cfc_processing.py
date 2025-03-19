import os
import json
import pandas as pd
import numpy as np
import mne
import pywt
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import cv2
from scipy.signal import iirnotch, butter, filtfilt
from constants import *

#IMPORTANT: run from root folder of repo so that cwd is AlzheimersDetection/
label_path = "data/participants.tsv"

def get_subject_group(subject_str):
    participant_id = "sub-" + subject_str
    participants_df = pd.read_csv(label_path, sep="\t")
    participant_row = participants_df[participants_df['participant_id'] == participant_id]
    assert(not participant_row.empty)
    return participant_row['Group'].values[0]

def get_cwt_scales(min_freq, max_freq, step_size, sampling_period_seconds):
    frequencies = np.arange(min_freq, max_freq, step_size)
    scales = 1 / (frequencies * sampling_period_seconds)
    return scales

def get_phase_bin(phase_rads):
    phase_rads = np.where(phase_rads == math.pi, -math.pi, phase_rads)
    assert np.all((phase_rads >= -math.pi) & (phase_rads < math.pi))
    bin_indices = ((phase_rads + math.pi) / (2 * math.pi) * N_PHASE_BINS).astype(int)
    assert np.all((bin_indices >= 0) & (bin_indices < N_PHASE_BINS))
    return bin_indices

def plot_phases_and_amplitudes(phases, amplitudes, alpha_scales, gamma_scales):
    # Heatmap for phases
    plt.subplot(2, 1, 1)
    plt.title("Alpha Frequency Phases (Heatmap)")
    plt.imshow(phases, aspect='auto', cmap='twilight', extent=[0, SEGMENT_LENGTH_SAMPLES, alpha_scales[-1], alpha_scales[0]])
    plt.colorbar(label='Phase (radians)')
    plt.xlabel("Time (samples)")
    plt.ylabel("Scales")

    # Heatmap for amplitudes
    plt.subplot(2, 1, 2)
    plt.title("Gamma Frequency Amplitudes (Heatmap)")
    plt.imshow(amplitudes, aspect='auto', cmap='viridis', extent=[0, SEGMENT_LENGTH_SAMPLES, gamma_scales[-1], gamma_scales[0]])
    plt.colorbar(label='Amplitude')
    plt.xlabel("Time (samples)")
    plt.ylabel("Scales")

    plt.tight_layout()
    plt.show()

def plot_signal_and_cwt_coefs(segment_channel, alpha_coefs, gamma_coefs):
    # Plot signal, alpha_coefs, and gamma_coefs
    plt.figure(figsize=(12, 12))
    # Plot the signal
    t = np.linspace(0, int(SEGMENT_LENGTH_SAMPLES/SAMPLING_FREQUENCY), SEGMENT_LENGTH_SAMPLES)
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
        extent=[t[0], t[-1], N_GAMMA, 0]
    )
    plt.colorbar(label="Magnitude")
    plt.title("Gamma Coefficients Heatmap")
    plt.xlabel("Time (s)")
    plt.ylabel("Scale")
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_gpac(global_pac_mi, subject_group, subject_str, segment_id):
    # plot the global pac
    aligned_global_pac_mi = global_pac_mi.T #no need to flip
    y_indices = np.arange(aligned_global_pac_mi.shape[0])
    x_indices = np.arange(aligned_global_pac_mi.shape[1])
    y = GAMMA_FREQUENCIES[0] + GAMMA_FREQUENCIES[2] * y_indices[:, np.newaxis]  # Make y a column vector for broadcasting
    x = ALPHA_FREQUENCIES[0] + ALPHA_FREQUENCIES[2] * x_indices  # Keep x as a row vector

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x, y, aligned_global_pac_mi, shading='auto', cmap='viridis')  # Color-coded representation
    plt.colorbar(label='Value')  # Add a color bar to indicate the scale
    plt.title(f'Color Plot for subject {subject_group} {subject_str} at t = {segment_id}')  # Title of the plot
    plt.xlabel('Phase Frequencies (Hz)')  # X-axis label
    plt.ylabel('Amplitude Frequencies (Hz)')  # Y-axis label
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()  # Display the plot

def plot_signal(signal, title="Signal Plot", xlabel="Sample Index", ylabel="Amplitude"):
    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D NumPy array.")
    
    plt.figure(figsize=(10, 4))
    plt.plot(signal, color='blue', linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_segment(segment, sampling_rate, channel_labels=CHANNEL_NAMES, title="EEG Signals"):
    if segment.shape[0] != 19:
        raise ValueError("Input data must have 19 channels (19xN array).")

    num_channels, num_samples = segment.shape
    time = np.linspace(0, num_samples / sampling_rate, num_samples)  # Time vector

    # Default channel labels if none are provided
    if channel_labels is None:
        channel_labels = [f"Ch{i+1}" for i in range(num_channels)]

    # Normalize signals for better visualization
    eeg_data_normalized = segment / np.max(np.abs(segment), axis=1, keepdims=True)

    # Plot each channel with vertical offsets
    plt.figure(figsize=(12, 8))
    offsets = np.arange(num_channels) * 2  # Offset each signal for clarity
    for i in range(num_channels):
        plt.plot(time, eeg_data_normalized[i] + offsets[i], label=channel_labels[i])

    # Plot settings
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Channels")
    plt.yticks(offsets, channel_labels)  # Place y-ticks at offsets with channel labels
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()

def interp(signal, sampling_freq, factor=2, filter_order=CFC_GAMMA_INTERP_LP_ORDER):
    # Step 1: Zero-padding
    upsampled_length = len(signal) * factor
    upsampled_signal = np.zeros(upsampled_length)
    upsampled_signal[::factor] = signal  # Insert the original signal at every factor-th position
    
    # Step 2: Design a low-pass filter
    nyquist = sampling_freq / 2  # Original Nyquist frequency
    cutoff = nyquist  # Default to Nyquist frequency of the original signal
    normalized_cutoff = cutoff / (factor * nyquist)  # Normalize cutoff for upsampled Nyquist frequency
    b, a = butter(filter_order, normalized_cutoff, btype='low')
    
    # Step 3: Filter the signal
    filtered_signal = filtfilt(b, a, upsampled_signal)  # Apply zero-phase filtering
    
    return filtered_signal

'''
def generate_synthetic_signal(
    sampling_frequency,
    duration_seconds,
    alpha_frequency,
    gamma_frequency,
    modulation_depth=0.5
):
    """
    Generate a synthetic EEG-like signal with PAC.

    Parameters:
    - sampling_frequency: int, Sampling frequency in Hz.
    - duration_seconds: float, Duration of the signal in seconds.
    - alpha_frequency: float, Frequency of the low-frequency phase component in Hz.
    - gamma_frequency: float, Frequency of the high-frequency amplitude component in Hz.
    - modulation_depth: float, Amplitude modulation depth (0 to 1).

    Returns:
    - signal: np.ndarray, Generated synthetic signal.
    """
    t = np.linspace(0, duration_seconds, int(sampling_frequency * duration_seconds), endpoint=False)

    # Generate alpha phase signal
    alpha_phase = np.sin(2 * np.pi * alpha_frequency * t)

    # Modulate gamma amplitude
    gamma_carrier = np.sin(2 * np.pi * gamma_frequency * t)
    modulated_gamma = (1 + modulation_depth * alpha_phase) * gamma_carrier

    return modulated_gamma

def verify_cfc_script():
    # Synthetic signal parameters
    duration = 2  # seconds
    sampling_frequency = SAMPLING_FREQUENCY # 500 Hz
    alpha_frequency = 6  # Hz (in the alpha band)
    gamma_frequency = 80  # Hz (in the gamma band)
    modulation_depth = 0.5  # Depth of amplitude modulation

    # Generate synthetic EEG-like signal
    synthetic_signal = generate_synthetic_signal(
        sampling_frequency, duration, alpha_frequency, gamma_frequency, modulation_depth
    )

    # Plot the synthetic signal
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, duration, len(synthetic_signal)), synthetic_signal)
    plt.title("Synthetic EEG-like Signal with PAC")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # Compute CWT scales for alpha and gamma bands
    alpha_scales = get_cwt_scales(ALPHA_FREQUENCIES[0], ALPHA_FREQUENCIES[1], ALPHA_FREQUENCIES[2])
    gamma_scales = get_cwt_scales(GAMMA_FREQUENCIES[0], GAMMA_FREQUENCIES[1], GAMMA_FREQUENCIES[2])

    # Apply CWT to extract coefficients
    alpha_coefs, _ = pywt.cwt(synthetic_signal, alpha_scales, f"cmor{CWT_B}-{CWT_C}", sampling_period=SAMPLING_PERIOD_SECONDS)
    gamma_coefs, _ = pywt.cwt(synthetic_signal, gamma_scales, f"cmor{CWT_B}-{CWT_C}", sampling_period=SAMPLING_PERIOD_SECONDS)

    # Extract phases (alpha) and amplitudes (gamma)
    alpha_phase, _ = extract_phase_amplitude(alpha_coefs)
    _, gamma_amplitude = extract_phase_amplitude(gamma_coefs)

    # Ensure both arrays have the same time dimension
    min_time_points = min(alpha_phase.shape[1], gamma_amplitude.shape[1])
    alpha_phase = alpha_phase[:, :min_time_points]
    gamma_amplitude = gamma_amplitude[:, :min_time_points]

    # Initialize PAC matrix
    pac_matrix = np.zeros((len(alpha_scales), len(gamma_scales)))

    # Calculate PAC matrix
    for a_idx, theta_phase in enumerate(alpha_phase):
        for g_idx, gamma_amp in enumerate(gamma_amplitude):
            pac_dist = calculate_pac(theta_phase, gamma_amp)
            pac_matrix[a_idx, g_idx] = calculate_modulation_index(pac_dist)

    # Flip PAC matrix for proper visualization
    flipped_pac_matrix = np.flip(pac_matrix.T, axis=0)

    # Plot PAC matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(
        flipped_pac_matrix,
        extent=[ALPHA_FREQUENCIES[0], ALPHA_FREQUENCIES[1], GAMMA_FREQUENCIES[0], GAMMA_FREQUENCIES[1]],
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    plt.colorbar(label="PAC Modulation Index")
    plt.title("PAC Matrix - Synthetic Signal")
    plt.xlabel("Theta Phase Frequency (Hz)")
    plt.ylabel("Gamma Amplitude Frequency (Hz)")
    plt.show()

    # Plot phases and amplitudes
    plot_phases_and_amplitudes(alpha_phase, gamma_amplitude, alpha_scales, gamma_scales)

    print("CFC script verification complete.")
'''

#global_epoch_id is the epoch_id across both datasets, used for naming
def process_epoch(global_epoch_id, is_healthy, subject_str, epoch, n_segments, segment_length_samples, alpha_scales, gamma_scales, sampling_freq):
    
    '''
    if global_epoch_id <= 324:
        return
    #TODO test with sine wave
    '''
    
    gpac_grads = np.empty((0, N_ALPHA, N_GAMMA))

    for segment_id in range(n_segments):
        start_sample = segment_id * STEP_LENGTH_SAMPLES
        end_sample = start_sample + segment_length_samples #exclusive
        segment = epoch[:, start_sample:end_sample]
        #print("global epoch: ", global_epoch_id, "\tsegment id: ", segment_id, "\tsegment shape: ", segment.shape)
        assert segment.shape == (N_CHANNELS, segment_length_samples)
        global_pac_mi = np.zeros((N_ALPHA, N_GAMMA))

        for channel_id in range(N_CHANNELS):
            segment_channel = segment[channel_id, :]
            assert segment_channel.shape == (segment_length_samples,)

            #calculate W(s, t) matrix
            #coefs, freqs = pywt.cwt(segment_channel, cwt_scales, f"cmor{CWT_B}-{CWT_C}")
            #alpha_coefs = coefs[0:N_ALPHA, :]
            #gamma_coefs = coefs[N_ALPHA:, :]
            
            #notch filter
            b_notch, a_notch = iirnotch(w0=CFC_NOTCH_FREQUENCY / (sampling_freq / 2), Q=CFC_NOTCH_QUALITY_FACTOR)
            segment_channel_notch = filtfilt(b_notch, a_notch, segment_channel)

            '''
            #interp for gamma bandpass filter
            interpolated_sampling_freq = sampling_freq
            if subject_str == "OSF":
                segment_channel_notch = interp(segment_channel_notch, sampling_freq)
                interpolated_sampling_freq *= 2
                plot_signal(segment_channel_notch)
            '''

            #filter to the alpha and gamma ranges
            b_alpha, a_alpha = butter(CFC_BAND_ORDER, [ALPHA_FREQUENCIES[0] / (0.5 * sampling_freq), ALPHA_FREQUENCIES[1] / (0.5 * sampling_freq)], btype='band')
            segment_channel_alpha = filtfilt(b_alpha, a_alpha, segment_channel_notch)
            #this fails for OSF bc upper limit of bandpass range exceeds Nyquist freq (64Hz)
            b_gamma, a_gamma = butter(CFC_BAND_ORDER, [GAMMA_FREQUENCIES[0] / (0.5 * sampling_freq), min(GAMMA_FREQUENCIES[1], sampling_freq / 2 - 1e-6) / (0.5 * sampling_freq)], btype='band')
            segment_channel_gamma = filtfilt(b_gamma, a_gamma, segment_channel_notch)

            #extract alpha freq phases
            alpha_coefs, _ = pywt.cwt(data=segment_channel_alpha, scales=alpha_scales, wavelet=f"cmor{CWT_B}-{CWT_C}")
            phases = np.angle(alpha_coefs)
            assert((phases <= math.pi).all() and (phases >= -math.pi).all())

            #extract gamma freq amplitudes
            gamma_coefs, _ = pywt.cwt(data=segment_channel_gamma, scales=gamma_scales, wavelet=f"cmor{CWT_B}-{CWT_C}")
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
            assert(pac_distributions.shape == pac_bin_mean_amplitudes.shape)
            pac_entropies = -np.sum(pac_distributions * np.log2(pac_distributions), axis=-1)
            pac_mi = (H_MAX - pac_entropies) / H_MAX
            assert np.all((pac_mi >= 0) & (pac_mi <= 1))
            global_pac_mi += pac_mi
        global_pac_mi /= N_CHANNELS

        #plot_segment(segment, sampling_freq)
        #plot_gpac(global_pac_mi, "CN" if is_healthy else "AD", subject_str, segment_id)

        #gradient
        grad_x = cv2.Sobel(global_pac_mi, cv2.CV_64F, dx=1, dy=0, ksize=3)  # Gradient in x-direction
        grad_y = cv2.Sobel(global_pac_mi, cv2.CV_64F, dx=0, dy=1, ksize=3)  # Gradient in y-direction
        grad = (grad_x * grad_x + grad_y * grad_y) ** 0.5
        assert(grad.shape == (N_ALPHA, N_GAMMA))
        gpac_grads = np.concatenate((gpac_grads, grad[np.newaxis, :, :]), axis=0)

    #standardization across all gpac gradients
    gpac_grad_mean = np.mean(gpac_grads)
    gpac_grad_std = np.std(gpac_grads)
    gpac_grads_standardized = (gpac_grads - gpac_grad_mean) / gpac_grad_std
    assert(gpac_grads_standardized.shape == (n_segments, N_ALPHA, N_GAMMA))

    #store standardized gPAC gradient (with tag) somewhere for later use
    epoch_path = f"data/cfc_overlap/cfc_{global_epoch_id}_{'cn' if is_healthy else 'ad'}.npy"
    np.save(epoch_path, gpac_grads_standardized)
    print(f"saved {epoch_path}")


def main():
    start_time = time.time()

    nemar_alpha_scales = get_cwt_scales(ALPHA_FREQUENCIES[0], ALPHA_FREQUENCIES[1], ALPHA_FREQUENCIES[2], SAMPLING_PERIOD_SECONDS)
    nemar_gamma_scales = get_cwt_scales(GAMMA_FREQUENCIES[0], GAMMA_FREQUENCIES[1], GAMMA_FREQUENCIES[2], SAMPLING_PERIOD_SECONDS)
    nemar_cwt_scales = np.concatenate((nemar_alpha_scales, nemar_gamma_scales))
    osf_alpha_scales = get_cwt_scales(ALPHA_FREQUENCIES[0], ALPHA_FREQUENCIES[1], ALPHA_FREQUENCIES[2], OSF_SAMPLING_PERIOD_SECONDS)
    osf_gamma_scales = get_cwt_scales(GAMMA_FREQUENCIES[0], GAMMA_FREQUENCIES[1], GAMMA_FREQUENCIES[2], OSF_SAMPLING_PERIOD_SECONDS)
    osf_cwt_scales = np.concatenate((osf_alpha_scales, osf_gamma_scales))
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
        #temp_data = subject_raw.get_data()[:, 654*SAMPLING_FREQUENCY:687*SAMPLING_FREQUENCY]
        #plot_segment(temp_data, SAMPLING_FREQUENCY)

        #filtering; NAH, using scipy now
        #subject_raw.notch_filter(freqs=[60])
        #subject_raw.filter(l_freq=None, h_freq=50)

        subject_data = subject_raw.get_data()
        assert(subject_data.shape == (N_CHANNELS, int(SAMPLING_FREQUENCY * recording_duration)))

        d_epoch_start_time_sec = (recording_duration - NEMAR_START_PADDING_LENGTH_SECONDS - EPOCH_LENGTH_SECONDS) / (N_EPOCHS - 1)
        d_epoch_start_time_sample = int(d_epoch_start_time_sec * SAMPLING_FREQUENCY)

        for epoch_id in range(N_EPOCHS):
            start_sample = NEMAR_START_PADDING_LENGTH_SAMPLES + epoch_id * d_epoch_start_time_sample
            epoch = subject_data[:, start_sample : (start_sample + EPOCH_LENGTH_SAMPLES)]
            assert(epoch.shape == (N_CHANNELS, EPOCH_LENGTH_SAMPLES))

            #print("subject id: ", subject_id, "\tglobal epoch: ", global_epoch_id)
            if global_epoch_id >= 800:
                process_epoch(global_epoch_id, subject_group == "C", subject_str, epoch, N_SEGMENTS, SEGMENT_LENGTH_SAMPLES, nemar_alpha_scales, nemar_gamma_scales, SAMPLING_FREQUENCY)
            global_epoch_id += 1
    
    '''
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
        assert(epoch.shape == (N_CHANNELS, OSF_SAMPLE_LENGTH_SAMPLES))

        #print("alz id: ", alz_id)
        process_epoch(global_epoch_id, False, "OSF", epoch, OSF_N_SEGMENTS, OSF_SEGMENT_LENGTH_SAMPLES, osf_alpha_scales, osf_gamma_scales, OSF_SAMPLING_FREQUENCY)
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
        assert(epoch.shape == (N_CHANNELS, OSF_SAMPLE_LENGTH_SAMPLES))

        process_epoch(global_epoch_id, True, "OSF", epoch, OSF_N_SEGMENTS, OSF_SEGMENT_LENGTH_SAMPLES, osf_alpha_scales, osf_gamma_scales, OSF_SAMPLING_FREQUENCY)
        global_epoch_id += 1
    '''
    
    end_time = time.time()
    print(f"time spent: {end_time - start_time}")


if __name__ == "__main__":
    main()