import os
import json
import pandas as pd
import numpy as np
import mne
import pywt
import math
import matplotlib.pyplot as plt
import time
import cv2
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

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(x, y, aligned_global_pac_mi, shading='auto', cmap='viridis')  # Color-coded representation
    plt.colorbar(label='Value')  # Add a color bar to indicate the scale
    plt.title(f'Color Plot for subject {subject_group} {subject_str} at t = {segment_id}')  # Title of the plot
    plt.xlabel('Phase Frequencies (Hz)')  # X-axis label
    plt.ylabel('Amplitude Frequencies (Hz)')  # Y-axis label
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

#global_epoch_id is the epoch_id across both datasets, used for naming
def process_epoch(global_epoch_id, is_healthy, epoch, n_segments, segment_length_samples, cwt_scales):
    gpac_grads = np.empty((0, N_ALPHA, N_GAMMA))
    
    for segment_id in range(n_segments):
        start_sample = segment_id * segment_length_samples
        end_sample = start_sample + segment_length_samples #exclusive
        segment = epoch[:, start_sample:end_sample]
        #print("global epoch: ", global_epoch_id, "\tsegment id: ", segment_id, "\tsegment shape: ", segment.shape)
        assert segment.shape == (N_CHANNELS, segment_length_samples)
        global_pac_mi = np.zeros((N_ALPHA, N_GAMMA))

        for channel_id in range(N_CHANNELS):
            segment_channel = segment[channel_id, :]
            assert segment_channel.shape == (segment_length_samples,)

            #calculate W(s, t) matrix
            coefs, freqs = pywt.cwt(segment_channel, cwt_scales, f"cmor{CWT_B}-{CWT_C}")
            alpha_coefs = coefs[0:N_ALPHA, :]
            gamma_coefs = coefs[N_ALPHA:, :]

            #extract alpha freq phases
            #alpha_coefs, _ = pywt.cwt(data=segment_channel, scales=alpha_scales, wavelet=f"cmor{CWT_B}-{CWT_C}")
            phases = np.angle(alpha_coefs)
            assert((phases <= math.pi).all() and (phases >= -math.pi).all())

            #extract gamma freq amplitudes
            #gamma_coefs, _ = pywt.cwt(data=segment_channel, scales=gamma_scales, wavelet=f"cmor{CWT_B}-{CWT_C}")
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

        #gradient
        grad_x = cv2.Sobel(global_pac_mi, cv2.CV_64F, dx=1, dy=0, ksize=3)  # Gradient in x-direction
        grad_y = cv2.Sobel(global_pac_mi, cv2.CV_64F, dx=0, dy=1, ksize=3)  # Gradient in y-direction
        grad = (grad_x * grad_x + grad_y * grad_y) ** 0.5
        assert(grad.shape == (N_ALPHA, N_GAMMA))
        gpac_grads = np.concatenate((gpac_grads, grad[np.newaxis, :, :]), axis=0)

    #standardization across all grac gradients
    gpac_grad_mean = np.mean(gpac_grads)
    gpac_grad_std = np.std(gpac_grads)
    gpac_grads_standardized = (gpac_grads - gpac_grad_mean) / gpac_grad_std
    assert(gpac_grads_standardized.shape == (n_segments, N_ALPHA, N_GAMMA))

    #store standardized gPAC gradient (with tag) somewhere for later use
    epoch_path = f"data/cfc/cfc_{global_epoch_id}_{'cn' if is_healthy else 'ad'}.npy"
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

        #filtering
        #subject_raw.notch_filter(freqs=[60])
        #subject_raw.filter(l_freq=None, h_freq=50)

        subject_data = subject_raw.get_data()
        assert(subject_data.shape == (N_CHANNELS, int(SAMPLING_FREQUENCY_HZ * recording_duration)))

        d_epoch_start_time_sec = (recording_duration - 2 * NEMAR_PADDING_LENGTH_SECONDS - EPOCH_LENGTH_SECONDS) / (N_EPOCHS - 1)
        d_epoch_start_time_sample = int(d_epoch_start_time_sec * SAMPLING_FREQUENCY_HZ)

        for epoch_id in range(N_EPOCHS):
            start_sample = NEMAR_PADDING_LENGTH_SAMPLES + epoch_id * d_epoch_start_time_sample
            epoch = subject_data[:, start_sample : (start_sample + EPOCH_LENGTH_SAMPLES)]
            assert(epoch.shape == (N_CHANNELS, EPOCH_LENGTH_SAMPLES))

            #print("subject id: ", subject_id, "\tglobal epoch: ", global_epoch_id)
            process_epoch(global_epoch_id, subject_group == "C", epoch, N_SEGMENTS, SEGMENT_LENGTH_SAMPLES, nemar_cwt_scales)
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
        assert(epoch.shape == (N_CHANNELS, OSF_SAMPLE_LENGTH_SAMPLES))

        print("alz id: ", alz_id)
        process_epoch(global_epoch_id, False, epoch, OSF_N_SEGMENTS, OSF_SEGMENT_LENGTH_SAMPLES, osf_cwt_scales)
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

        process_epoch(global_epoch_id, True, epoch, OSF_N_SEGMENTS, OSF_SEGMENT_LENGTH_SAMPLES, osf_cwt_scales)
        global_epoch_id += 1
    
    end_time = time.time()
    print(f"time spent: {end_time - start_time}")


if __name__ == "__main__":
    main()