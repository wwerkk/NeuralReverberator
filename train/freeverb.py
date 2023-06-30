import numpy as np
import soundfile as sf
import random

# Comb filter
def comb_filter(x, delay, feedback, damp, undamp):
    buffer = np.zeros(delay)
    output = np.zeros_like(x)
    filterstore = 0
    bufidx = 0

    for i in range(len(x)):
        input = x[i]
        output[i] = buffer[bufidx]
        filterstore = (output[i]*damp) + (filterstore*undamp)  # lowpass filter
        buffer[bufidx] = input + (filterstore * feedback)
        if bufidx == 0:
            bufidx = delay - 1
        else:
            bufidx -= 1

    return output

# Allpass filter
def allpass_filter(x, delay, feedback):
    buffer = np.zeros(delay)
    output = np.zeros_like(x)
    bufidx = 0

    for i in range(len(x)):
        input = x[i]
        bufout = buffer[bufidx]
        output[i] = -input + bufout
        buffer[bufidx] = input + (bufout*feedback)
        if bufidx == 0:
            bufidx = delay - 1
        else:
            bufidx -= 1

    return output

# Freeverb
def freeverb(x, n_c=4, comb_feedback=0.84, comb_damping=0.2,
             n_a=4, allpass_feedback=0.5):
    comb_delays = []
    for i in range(n_c):
        comb_delays.append(random.randint(100, 2000))
    allpass_delays = []
    for i in range(n_a):
        allpass_delays.append(random.randint(100, 2000))
    
    # Apply comb filters in parallel
    y = sum(comb_filter(x, delay, comb_feedback, comb_damping, 1-comb_damping) for delay in comb_delays)
    
    # Apply allpass filters in series
    for delay in allpass_delays:
        y = allpass_filter(y, delay, allpass_feedback)
    
    return y

# Apply Freeverb to an audio file
def apply_freeverb_to_audio_file(input_filename, output_filename):
    # Load audio file
    x, samplerate = sf.read(input_filename)
    
    # Apply Freeverb
    y = freeverb(x)
    
    # Write output audio file
    sf.write(output_filename, y, samplerate)

# Example usage
# apply_freeverb_to_audio_file('input.wav', 'output.wav')