import numpy as np
import soundfile as sf
from keras.models import load_model
from util import load_specgrams, load_params
from os import getcwd, chdir, makedirs
from soundfile import write
from dsp_ import prime, simple_fdn
from util import generate_specgram, plot_specgram
from scipy.linalg import hadamard

FDN_SIZE = 16
SAMPLE_RATE = 16000
MAX_LENGTH = 1000
PLOT = True

# Set the paths
p_dir = "data_/params"
spect_dir = "data_/spectrograms"

gp_dir = "generated/params"
gy_dir = "generated/audio"
gspect_dir = "generated/spectrograms"
gplot_dir = "generated/spect_plots"

# Change the working directory to the train folder
if getcwd().split('/')[-1] != 'train':
    chdir('train')

# Create the output directory if it doesn't exist
makedirs(gp_dir, exist_ok=True)
makedirs(gy_dir, exist_ok=True)
makedirs(gspect_dir, exist_ok=True)
makedirs(gplot_dir, exist_ok=True)

model = load_model("reports/train_2023_11_12__23-17/regression.hdf5")
# print(model.summary())
specgrams, _ = load_specgrams('data_/spectrograms', (513, 128), train_split=1.0)
print(specgrams.shape)
pred = model.predict(specgrams)
params = load_params('data_/params', train_split=1.0)

# Unipolar impulse
x = np.zeros(2)
x[0] = -1
x[1] = 1

H = hadamard(FDN_SIZE) * 0.25

PRIME_LIST = prime(0, 30000)

for i, p in enumerate(params[0]):
    p = params[0][0]
    p_ = pred[i]
    print(f"Truth:")
    print(p)
    print(f"Predictions:")
    print(p_)
    y = simple_fdn(x,
                   decay=p_[0],
                   min_dist=p_[1],
                   max_dist=p_[2],
                   distance_curve=p_[3],
                   min_freq=p_[4],
                   max_freq=p_[5],
                   frequency_curve=p_[6],
                   H=H,
                   prime_list=PRIME_LIST.astype(np.int32),
                   sr=SAMPLE_RATE,
                   max_length=MAX_LENGTH)
    # Save impulse response audio
    gy_path = gy_dir + '/' + "impulse" + f"_{i}.wav"
    write(gy_path, y, SAMPLE_RATE)
    # Save FDN params
    gp_path = gp_dir + '/' + "impulse" + f"_{i}.txt"
    np.savetxt(gp_path, p_)
    # Generate and save the spectrogram
    Sn = np.array(generate_specgram(y, SAMPLE_RATE))
    gspect_path = gspect_dir + '/' + "impulse" + f"_{i}.txt"
    np.savetxt(gspect_path, Sn)
    # Optional: Generate and save spectrogram plot
    if PLOT:
        plot_specgram(Sn, SAMPLE_RATE, "impulse" + f"_{i}", gplot_dir)
    
    print(f"{i+1}/{len(params[0])}")