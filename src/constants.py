import math


#gotten from raw.info['ch_names']
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
SEGMENT_LENGTH_SECONDS = 2

#from 6.2.2 and 6.4.1 of thesis
ALPHA_FREQUENCIES = (4, 8, 0.1)
#GAMMA_FREQUENCIES = (50, 80, 1)
GAMMA_FREQUENCIES = (55, 75, 1)
N_ALPHA = int((ALPHA_FREQUENCIES[1] - ALPHA_FREQUENCIES[0]) / ALPHA_FREQUENCIES[2])
N_GAMMA = int((GAMMA_FREQUENCIES[1] - GAMMA_FREQUENCIES[0]) / GAMMA_FREQUENCIES[2])

#from 6.2.1 of thesis
CWT_B = 6
CWT_C = 0.8125

#from ref [26] of thesis
N_PHASE_BINS = 18
H_MAX = math.log2(N_PHASE_BINS)

#filters
CFC_NOTCH_FREQUENCY = 60
CFC_NOTCH_QUALITY_FACTOR = 30
CFC_BAND_ORDER = 6
CFC_GAMMA_INTERP_LP_ORDER = 6

#for FFT
FFT_LP_FREQUENCY = 50
FFT_LP_ORDER = 6
N_FREQUENCY_BANDS = 5
FREQUENCY_BANDS = [0, 4, 8, 12, 30, 50]

#============= NEMAR =============
#gotten from https://nemar.org/dataexplorer/detail?dataset_id=ds004504
SAMPLING_FREQUENCY = 500
SAMPLING_PERIOD_SECONDS = 1 / SAMPLING_FREQUENCY
EPOCH_LENGTH_SECONDS = 12
EPOCH_LENGTH_SAMPLES = EPOCH_LENGTH_SECONDS * SAMPLING_FREQUENCY
NEMAR_START_PADDING_LENGTH_SECONDS = 5
NEMAR_START_PADDING_LENGTH_SAMPLES = NEMAR_START_PADDING_LENGTH_SECONDS * SAMPLING_FREQUENCY
N_CHANNELS = 19
N_AD = 36
N_FTD = 23
N_CN = 29
N_SUBJECTS = N_AD + N_FTD + N_CN
N_EPOCHS = 5
N_SEGMENTS = int(EPOCH_LENGTH_SECONDS / SEGMENT_LENGTH_SECONDS)
SEGMENT_LENGTH_SAMPLES = SEGMENT_LENGTH_SECONDS * SAMPLING_FREQUENCY

#============= OSF =============
#gotten from https://osf.io/2v5md/
#some details are gotten from the "Computational methods of EEG signals analysis for Alzheimer’s disease classification" paper
OSF_N_AD = 80
OSF_N_CN = 12
OSF_N_SUBJECTS = OSF_N_AD + OSF_N_CN
OSF_SAMPLING_FREQUENCY = 128
OSF_SAMPLING_PERIOD_SECONDS = 1 / OSF_SAMPLING_FREQUENCY
OSF_SAMPLE_LENGTH_SECONDS = 8
OSF_SAMPLE_LENGTH_SAMPLES = OSF_SAMPLE_LENGTH_SECONDS * OSF_SAMPLING_FREQUENCY
OSF_PADDING_LENGTH_SECONDS = 0 #osf appears to require no padding
OSF_PADDING_LENGTH_SAMPLES = OSF_PADDING_LENGTH_SECONDS * OSF_SAMPLING_FREQUENCY
OSF_N_SEGMENTS = int(OSF_SAMPLE_LENGTH_SECONDS / SEGMENT_LENGTH_SECONDS)
OSF_SEGMENT_LENGTH_SAMPLES = SEGMENT_LENGTH_SECONDS * OSF_SAMPLING_FREQUENCY
