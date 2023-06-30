import os
import soundfile as sf
import librosa
import random
import numpy as np
from freeverb import freeverb


input_file = 'train/data/dry/balloon_burst.wav'    
output_dir = 'train/data/balloon_verb/'
N  = 16 # num of samples to generate
len_s = 2 # length of samples in seconds

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Load audio file
x, samplerate = sf.read(input_file)
for i in range(N):
    # Apply Freeverb
    # c_d = random.random()
    c_d = 0.8
    c_fb = random.normalvariate(0.4, 0.3)
    a_fb = random.normalvariate(0.5, 0.4)
    y = freeverb(x, n_c=4, n_a=4, comb_damping=c_d, comb_feedback=c_fb, allpass_feedback=a_fb)
    # Write output audio file
    output_path = output_dir + (f'{i}.wav')
    sf.write(output_path, y, samplerate)
print(f'Processed: {input_file} {N} times')