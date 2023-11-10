import os
import numpy as np
from soundfile import read, write
from dsp import freeverb
import time
from resampy import resample
from librosa import stft, power_to_db
from librosa import stft, power_to_db, display
from matplotlib import pyplot as plt


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

def plot_specgram(log_power_spectra, rate, filename, output_dir):
    """ 
    Save log-power and normalized log-power specotrgram to file.

    Args:
        log_power_spectra (ndarray): Comptued Log-Power spectra.
        rate (int): Sample rate of input audio data.
        filename (str): Output filename for generated plot.
        output_dir (str): Directory to save generated plot.
    """

    plt.figure()
    display.specshow(log_power_spectra, sr=rate*2, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Power spectrogram')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close('all')


# Set the paths
x_file = "noise_burst.wav"
x_dir = "data/dry"
p_dir = "params"
y_dir = "audio"
spectrograms_dir = "spectrograms"
plots_dir = "plots"

n_samples = 50
length_s = 2
SR = 16000
plot = True

# Change the working directory to the train folder
if os.getcwd().split('/')[-1] != 'train':
    os.chdir('train')
 
# Create the output directory if it doesn't exist
os.makedirs(p_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)
os.makedirs(spectrograms_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Define the Freeverb parameter space
n_c = 8
c = (
    (1000, 500), # delay mean, stdev
    (0.3, 0.1), # feedback gain mean, stdev
    (0.7, 0.1) # damping mean, stdev
)

n_a = 4
a = (
    (300, 100), # delay mean, stdev
    (0.3, 0.2) # feedback gain mean, stdev
)

# Set the random seed for reproducibility
np.random.seed(42)

# Load the audio file
file_path = os.path.join(x_dir, x_file)
x, sr = read(file_path)
if sr != SR:
    x = resample(x, sr, SR)
    sr = SR
# Normalize the x to the range [-1, 1]
x /= max(abs(x))
length = length_s * sr

start_time = time.time()

P = [] # params
S = [] # spectrograms

for i in range(n_samples):
    # Apply the Freeverb effect with randomized peters
    cM = np.random.normal(c[0][0], c[0][1], n_c).astype(int)
    ca = np.random.normal(c[1][0], c[1][1], n_c)
    cd = np.random.normal(c[2][0], c[2][1], n_c)
    aM = np.random.normal(a[0][0], a[0][1], n_a).astype(int)
    aa = np.random.normal(a[1][0], a[1][1], n_a)

    # Clip gain params to prevent exploding feedback
    ca = np.clip(ca, 0, 1)
    cd = np.clip(cd, 0, 1)
    aa = np.clip(aa, 0, 1)

    # Process the audio file with the Freeverb effect
    y = freeverb(
        x=x,
        cM=cM,
        ca=ca,
        cd=cd,
        aM=aM,
        aa=aa
    )

    # Truncate to desired length
    y = y[:length]

    # Store the params in a .txt file
    p_filename = os.path.splitext(x_file)[0] + f"_{i}"
    p_path = os.path.join(p_dir, p_filename)
    p = np.concatenate([cM, ca, cd, aM, aa], dtype=object)
    np.savetxt(p_path, p, fmt='%.10f')

    # Save the processed audio to a new audio file
    y_filename = os.path.splitext(x_file)[0] + f"_{i}.wav"
    y_path = os.path.join(y_dir, y_filename)
    write(y_path, y, sr)
    print(f"Generated: {i+1}/{n_samples}")

    # Generate the spectrogram
    Sn = np.array(generate_specgram(y, sr))
    specgram_filename = os.path.splitext(x_file)[0] + f"_{i}"
    specgram_path = os.path.join(spectrograms_dir, specgram_filename)
    np.savetxt(specgram_path, Sn)

    if plot:
        plot_specgram(Sn, sr, specgram_filename, plots_dir)

print(f"{n_samples} samples generated.")

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")