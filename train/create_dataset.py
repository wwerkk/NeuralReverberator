import os
import numpy as np
from soundfile import read, write
from dsp import freeverb
import time
from resampy import resample
from librosa import stft, power_to_db


def generate_specgram(x, sr=16000, n_fft=1024, n_hop=256):
    """
    Generate a spectrogram (via stft) on input audio data.

    Args:
        x (ndarray): Input audio data.
        sr (int, optional): Sample rate out input audio data.
        n_fft (int, optional): Size of the FFT to generate spectrograms.
        n_hop (int, optional): Hop size for FFT.
    """
    S = stft(x, n_fft=n_fft, hop_length=n_hop, center=True)
    power_spectra = np.abs(S)**2
    log_power_spectra = power_to_db(power_spectra)
    _min = np.amin(log_power_spectra)
    _max = np.amax(log_power_spectra)
    if _min == _max:
        print(f"divide by zero in audio array")
    else:
        normalized_log_power_spectra = (log_power_spectra - _min) / (_max - _min)
    return normalized_log_power_spectra


# Set the paths
x_file = "noise_burst.wav"
x_dir = "data/dry"
p_dir = "params"
y_dir = "audio"
spectrograms_dir = "spectrograms"

n_samples = 500
length_s = 2
sr = 16000

# Change the working directory to the train folder
if os.getcwd().split('/')[-1] != 'train':
    os.chdir('train')
 
# Create the output directory if it doesn't exist
os.makedirs(p_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)
os.makedirs(spectrograms_dir, exist_ok=True)

## TODO: Parameter space and param generation for FDN reverberator
# "Decay"
# "Min distance"
# "Max distance"
# "Distance curve"
# "Low frequency" 
# "Frequency curve"
# "High frequency"

## TODO: Additionally - completely random delay lengths

# Set the random seed for reproducibility
np.random.seed(42)

## TODO: replace with generated impulse array
## TODO: get rid of the below code
# Load the audio file
# file_path = os.path.join(x_dir, x_file)
# x, sr = read(file_path)
# if sr != SR:
#     x = resample(x, sr, SR)
#     sr = SR
# Normalize the x to the range [-1, 1]
# x /= max(abs(x))
length = length_s * sr

start_time = time.time()

# P = [] # params
# S = [] # spectrograms

for i in range(n_samples):
    # Apply the Freeverb effect with randomized peters
    # cM = np.random.normal(c[0][0], c[0][1], n_c).astype(int)
    # ca = np.random.normal(c[1][0], c[1][1], n_c)
    # cd = np.random.normal(c[2][0], c[2][1], n_c)
    # aM = np.random.normal(a[0][0], a[0][1], n_a).astype(int)
    # aa = np.random.normal(a[1][0], a[1][1], n_a)

    # # Clip gain params to prevent exploding feedback
    # ca = np.clip(ca, 0, 1)
    # cd = np.clip(cd, 0, 1)
    # aa = np.clip(aa, 0, 1)

    # # Process the audio file with the Freeverb effect
    # y = freeverb(
    #     x=x,
    #     cM=cM,
    #     ca=ca,
    #     cd=cd,
    #     aM=aM,
    #     aa=aa
    # )

    # # Truncate to desired length
    # y = y[:length]

    # Store the params in a .txt file
    p_filename = os.path.splitext(x_file)[0] + f"_{i}.txt" # add Bell curve Wojack to slides
    p = np.concatenate([cM, ca, cd, aM, aa], dtype=object)
    np.savetxt(p_path, p, fmt='%.10f')

    # Save the processed audio to a new audio file
    y_filename = os.path.splitext(x_file)[0] + f"_{i}.wav"
    y_path = os.path.join(y_dir, y_filename)
    write(y_path, y, sr)
    print(f"Generated: {i+1}/{n_samples}")

    # Generate the spectrogram
    Sn = np.array(generate_specgram(y, sr))
    specgram_filename = os.path.splitext(x_file)[0] + f"_{i}.txt" # add Bell curve Wojack to slides
    specgram_path = os.path.join(spectrograms_dir, specgram_filename)
    np.savetxt(specgram_path, Sn)

    print(f"{i}/{n_samples}")

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")