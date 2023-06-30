import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from keras.models import load_model
from util import ispecgram, fix_specgram_shape, plot_specgrams
from generate import generate_z, generate_specgram, audio_from_specgram

import matplotlib as mpl  
mpl.use('agg')
import matplotlib.pyplot as plt

def process_spec(spec_dir):
    # loaded trained models
    encoder = load_model("train/models/encoder.hdf5")
    decoder = load_model("train/models/decoder.hdf5")
    name = spec_dir.split('/')[-1].split('.')[:-1][0]
    spec = np.loadtxt(spec_dir)
    print(spec.shape)
    spec = fix_specgram_shape(spec, (513, 128))
    plot_specgrams(spec, 16000, f'{name}_orig.png', './')
    spec = np.reshape(spec, (513, 128, 1))
    _z = generate_z(encoder, spec)
    print('z = {}'.format(_z))

    out_spec = generate_specgram(decoder, _z)
    print(out_spec.shape)
    plot_specgrams(out_spec, 16000, f'{name}_recon.png', './')
    audio_from_specgram(out_spec, 16000, f'{name}_recon')

process_spec('train/spectrograms/ir_0_16000_11.txt')