import math

SAMPLING_FREQUENCY_HZ = 500
SAMPLING_PERIOD_SECONDS = 1 / SAMPLING_FREQUENCY_HZ
EPOCH_LENGTH_SECONDS = 12
EPOCH_LENGTH_SAMPLES = EPOCH_LENGTH_SECONDS * SAMPLING_FREQUENCY_HZ
SUBJECT_PADDING_LENGTH_SECONDS = 4
SUBJECT_PADDING_LENGTH_SAMPLES = SUBJECT_PADDING_LENGTH_SECONDS * SAMPLING_FREQUENCY_HZ
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
ALPHA_FREQUENCIES = (3, 4.7, 0.05)
GAMMA_FREQUENCIES = (57, 79, 1)
N_ALPHA = int((ALPHA_FREQUENCIES[1] - ALPHA_FREQUENCIES[0]) / ALPHA_FREQUENCIES[2])
N_GAMMA = int((GAMMA_FREQUENCIES[1] - GAMMA_FREQUENCIES[0]) / GAMMA_FREQUENCIES[2])

#from ref [26] of thesis
N_PHASE_BINS = 18
H_MAX = math.log2(N_PHASE_BINS)