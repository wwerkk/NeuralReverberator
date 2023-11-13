import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import numpy as np
import scipy
import pandas as pd
from soundfile import write
import time
from dsp_ import prime, simple_fdn
import os
from util import generate_specgram, plot_specgram

# Set the paths
p_dir = "data_/params"
y_dir = "data_/audio"
spect_dir = "data_/spectrograms"
plot_dir = "data_/spect_plots"

# Change the working directory to the train folder
if os.getcwd().split('/')[-1] != 'train':
    os.chdir('train')

# Create the output directory if it doesn't exist
os.makedirs(p_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)
os.makedirs(spect_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# FDN size and samplerate
FDN_SIZE = 16
SAMPLE_RATE = 16000
IMPULSE_NUM = 10
MAX_LENGTH = 1000
PLOT = True

# Set the random seed for reproducibility
np.random.seed(42)

# Unipolar impulse
x = np.zeros(2)
x[0] = -1
x[1] = 1

# Hadamard matrix
H = scipy.linalg.hadamard(FDN_SIZE) * 0.25
start_time = time.time()

# Random parameter arrays
decay           = 0.06 + np.random.random((IMPULSE_NUM)) * (1 - 0.06)
min_dist        = 0.001 + np.random.random((IMPULSE_NUM)) * (0.1 - 0.001)
max_dist        = 0.1 + np.random.random((IMPULSE_NUM)) * 0.9
distance_curve  = np.random.random((IMPULSE_NUM))
min_freq        = np.random.random((IMPULSE_NUM)) * 0.5
max_freq        = 0.5 + np.random.random((IMPULSE_NUM)) * 0.5
frequency_curve = np.random.random((IMPULSE_NUM))

parameters = pd.DataFrame(
    {
        "Decay":           decay,
        "Min distance":    min_dist,
        "Max distance":    max_dist,
        "Distance curve":  distance_curve,
        "Low frequency":   min_freq,
        "High frequency":  max_freq,
        "Frequency curve": frequency_curve
    }
)

# prime number list for delay line
PRIME_LIST = prime(0, 30000)

for i in range(IMPULSE_NUM):
    print(parameters.values[i])
    y = simple_fdn(x,
                   decay=decay[i],
                   min_dist=min_dist[i],
                   max_dist=max_dist[i],
                   distance_curve=distance_curve[i],
                   min_freq=min_freq[i],
                   max_freq=max_freq[i],
                   frequency_curve=frequency_curve[i],
                   H=H,
                   prime_list=PRIME_LIST.astype(np.int32),
                   sr=SAMPLE_RATE,
                   max_length=MAX_LENGTH)

    # Save impulse response audio
    y_path = y_dir + '/' + "impulse" + f"_{i}.wav"
    write(y_path, y, SAMPLE_RATE)
    # Save FDN params
    p_path = p_dir + '/' + "impulse" + f"_{i}.txt"
    np.savetxt(p_path, parameters.values[i])
    # Generate and save the spectrogram
    Sn = np.array(generate_specgram(y, SAMPLE_RATE))
    spect_path = spect_dir + '/' + "impulse" + f"_{i}.txt"
    np.savetxt(spect_path, Sn)
    # Optional: Generate and save spectrogram plot
    if PLOT:
        plot_specgram(Sn, SAMPLE_RATE, "impulse" + f"_{i}", plot_dir)
    
    print(f"{i+1}/{IMPULSE_NUM}")

end_time = time.time()
runtime = end_time - start_time
print(f"Generated: {IMPULSE_NUM} RIRs in {runtime} seconds")