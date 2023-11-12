import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import numpy as np                          
import scipy
import pandas as pd
from soundfile import  write
from dsp_ import prime
import time
from dsp_ import simple_fdn
import os
from util import generate_specgram, plot_specgram
from multiprocessing import Pool

# Set the paths
p_dir = "data/params"
y_dir = "data/audio"
spect_dir = "data/spectrograms"
plot_dir = "data/spect_plots"

# FDN size and samplerate
FDN_SIZE = 16
SAMPLE_RATE = 16000
IMPULSE_NUM = 5
MAX_LENGTH = 1000
PLOT = True
NUM_WORKERS = os.cpu_count() # default to number of cores, can be set manually

# Change the working directory to the train folder
if os.getcwd().split('/')[-1] != 'train':
    os.chdir('train')

# Create the output directory if it doesn't exist
os.makedirs(p_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)
os.makedirs(spect_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Set the random seed for reproducibility
np.random.seed(42)

# Unipolar impulse
x = np.zeros(2)
x[0] = -1
x[1] = 1

# Hadamard matrix
H = scipy.linalg.hadamard(FDN_SIZE) * 0.25

# Random parameter arrays
decay           = np.random.random((IMPULSE_NUM))
min_dist        = np.random.random((IMPULSE_NUM)) * 0.1
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
PRIME_LIST = prime(0, 30000).astype(np.int32)

# 2 second impulse responses
IMPULSE_LENGTH = (SAMPLE_RATE * 2) + x.shape[-1]

def generate_impulse(i, params, H, PRIME_LIST, SAMPLE_RATE, MAX_LENGTH):
    y = simple_fdn(x,
                    decay=params[0],
                    min_dist=params[1],
                    max_dist=params[2],
                    distance_curve=params[3],
                    min_freq=params[4],
                    max_freq=params[5],
                    frequency_curve=params[6],
                    H=H,
                    prime_list=PRIME_LIST,
                    sr=SAMPLE_RATE,
                    max_length=MAX_LENGTH)
    Sn = np.array(generate_specgram(y, SAMPLE_RATE))
    return (i, y, Sn)


def main():
    start_time = time.time()

    with Pool(NUM_WORKERS) as pool:
        results = pool.starmap(generate_impulse, [(i, parameters.values[i], H, PRIME_LIST, SAMPLE_RATE, MAX_LENGTH) for i in range(IMPULSE_NUM)])

    for i, y, Sn in results:
    # Save impulse response audio
        y_path = y_dir + '/' + "impulse" + f"_{i}.wav"
        write(y_path, y, SAMPLE_RATE)
        # Save FDN params
        p_path = p_dir + '/' + "impulse" + f"_{i}"
        # print(parameters.values[i])
        np.savetxt(p_path, parameters.values[i])
        # Generate and save the spectrogram
        spect_path = spect_dir + '/' + "impulse" + f"_{i}"
        np.savetxt(spect_path, Sn)
        # Optional: Generate and save spectrogram plot
        if PLOT:
            plot_specgram(Sn, SAMPLE_RATE, "impulse" + f"_{i}", plot_dir)
        print(f"{i+1}/{IMPULSE_NUM}")

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Finished in: {runtime} seconds")

if __name__ == "__main__":
    main()