import numpy as np
import numpy.typing as npt
from numba import jit, njit, prange, vectorize, int32, int64, float32, float64

@njit(nopython=True)
def pad(a, pad_width, constant_values=0):
    """
    Pads a 1D NumPy array with a constant value.

    Parameters:
    a (ndarray): Input array.
    pad_width (int or sequence of ints): Number of values padded to the edges of each axis.
    constant_values (scalar or sequence): Values used for padding. Default is 0.

    Returns:
    ndarray: Padded array.

    Note:
    This function works only for 1D arrays.
    """
    a_ = np.zeros(a.shape[-1] + pad_width)
    a_[:a.shape[-1]] = a
    return a_

@njit(nopython=True)
def padadd(a, b):
    """
    Adds two 1D NumPy arrays, padding the shorter array if their lengths do not match.

    Parameters:
    a (ndarray): First input array.
    b (ndarray): Second input array.

    Returns:
    ndarray: Sum of the two arrays with padding.

    Note:
    This function works only for 1D arrays.
    """
    len_a = a.shape[-1]
    len_b = b.shape[-1]
    max_len = max(len_a, len_b)
    if len_a < max_len:
        a = pad(a, max_len - len_a)
    if len_b < max_len:
        b = pad(b, max_len - len_b)
    return a + b


@njit
def fbcf(x: npt.NDArray[np.float64], b: float = 1.0, M: int = 2000, a: float = 0.9) -> np.ndarray[np.float64]:
    """
    Implements a feedback comb filter.

    Args:
        x (np.ndarray): Input signal.
        b (float, optional): Input gain. Defaults to 1.0.
        M (int, optional): Delay time in samples. Defaults to 2000.
        a (float, optional): Feedback gain. Defaults to 0.9.

    Returns:
        np.ndarray: Output signal.
    """
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in range(y.shape[-1]):
        if i < x.shape[-1]:
            y[i] += b * x[i]
        if i >= M:
            y[i] += feedback
            feedback = -a * y[i - M]
    return y

@njit
def lbcf(x: npt.NDArray[np.float64], b: float = 1.0, M: int = 2000, a: float = 0.9, d: float = 0.5) -> np.ndarray[np.float64]:
    """
    Implements Schroeder's Lowpass-Feedback Comb Filter.

    Args:
        x (np.ndarray): Input signal.
        b (float, optional): Input gain. Defaults to 1.0.
        M (int, optional): Delay time in samples. Defaults to 2000.
        a (float, optional): Feedback gain. Defaults to 0.9.
        d (float, optional): Damping factor. Defaults to 0.5.

    Returns:
        np.ndarray: Output signal.
    """
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in range(y.shape[-1]):
        if i < x.shape[-1]:
            y[i] += b * x[i]
        if i >= M:
            y[i] += feedback
            feedback += (1 - d) * ((a * y[i - M]) - feedback)
    return y

@njit
def allpass(x: npt.NDArray[np.float64], M: int = 2000, a: float = 0.5) -> npt.NDArray[np.float64]:
    """
    Implements an allpass filter.

    Args:
        x (np.float32): Input signal.
        M (int, optional): Delay time in samples. Defaults to 2000.
        a (float, optional): Feedback gain. Defaults to 0.5.

    Returns:
        np.float32: Output signal.
    """
    feedback = 0
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in range(y.shape[-1]):
        if i < x.shape[-1]:
            y[i] = x[i] - feedback
            feedback *= a
            if i >= M:
                feedback += x[i - M]
        else:
            y[i] -= feedback
            feedback *= a
    return y

def calculate_feedback_gain(T60, M):
    """
    Calculate the feedback gain (a) for a Schroeder comb filter based on the given T60 decay time and delay length (M).
    
    Args:
        T60 (float): Desired decay time in seconds.
        M (int): Delay length in samples.
    
    Returns:
        float: Calculated feedback gain.
    """
    return np.exp((np.log(0.001) * M) / T60)

@njit
def freeverb(
        x,
        cb=np.full(8, 1.0),
        cM=np.array([1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116], dtype=np.int64),
        ca=np.full(8, 0.84),
        cd=np.full(8, 0.2),
        aM=np.array([225, 556, 441, 341], dtype=np.int64),
        aa=np.full(4, 0.5)
        ):
    """
    Applies Schroeder reverberator algorithm (aka Freeverb) to the input signal.
    https://ccrma.stanford.edu/~jos/pasp/Freeverb.html

    Args:
        x (np.ndarray): Input signal.
        cb (optional): List of input gains for parallel lowpass-feedback comb filters. Defaults to [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].
        cM (optional): List of delay times in samples for parallel lowpass-feedback comb filters. Defaults to [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116].
        ca (optional): List of feedback gains for parallel lowpass-feedback comb filters. Defaults to [0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84].
        cd (optional): List of damping factors for parallel lowpass-feedback comb filters. Defaults to [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2].
        aM (optional): List of delay times in samples for cascading allpass filters. Defaults to [225, 556, 441, 341].
        aa (optional): List of feedback gains for cascading allpass filters. Defaults to [0.5, 0.5, 0.5, 0.5].

    Returns:
        np.ndarray: Output signal.
    """
    ca = np.clip(ca, 1.0714, 0.98) # clip range to stable values
    y = np.zeros(x.shape[-1])
    # Apply paralell low-passed feedback comb filters
    for b, M, a, d in zip(cb, cM, ca, cd):
        y_ = lbcf(x=x, b=b, M=M, a=a, d=d)
        shape = y.shape[-1]
        shape_ = y_.shape[-1]
        if shape < shape_:
            y = pad(y, shape_-shape)
        elif shape > shape_:
            y_ = pad(y_, shape-shape_)
        y += y_
    # Apply cascading allpass filters
    for M, a in zip(aM, aa):
        y = allpass(y, M, a)

    max_abs_value = np.max(np.abs(y))
    epsilon = 1e-12
    y = y / (max_abs_value + epsilon)
    return y


@jit("float64[:](float64[:], int32, boolean, float64)", nopython=True, fastmath=False)
def feedforward_delay(x, loop_time=1, milliseconds=False, sr=44100):
    
    M = (loop_time / 1000) * sr if milliseconds else loop_time
    M = int(M) if (M > 0) else 1
    
    d = np.zeros(M) # wrap this buffer and add last value to input array.
    p = 0 # delay buffer index
    y = np.zeros(x.size) # output buffer
    
    for i in range(x.size):
        
        # remove one modulo operation.
        # p = p % M
        
        y[i] = d[p % M] 
        d[p % M] = x[i]
        p += 1
        
    return y

@jit("float64[:](float64[:], float64, float64)", nopython=True, fastmath=False)
def onepole(input, freq=200, sr=44100):
    
    output = np.zeros((input.shape[-1]))
    
    x = np.exp(-2.0*np.pi*freq/sr)
    
    a0 = 1-x
    b1 = -x
    
    tmp = 0
    
    for i in range(len(output)):
        output[i] = a0 * input[i] - b1*tmp
        tmp = output[i]
        
    return output

@jit("float64[:](float64[:], float64)", nopython=True, fastmath=False)
def dc_block(input, sr=44100):
    
    freq = 20
    output = np.zeros((input.shape[-1]))
    
    x = np.exp(-2.0*np.pi*freq/sr)
    
    a0 = 1-x
    b1 = -x
    
    tmp = 0
    
    for i in range(len(output)):
        output[i] = a0 * input[i] - b1*tmp
        tmp = output[i]
        
    return input - output

def prime(x, y):
    prime_list = []
    for i in range(x, y):
        if i == 0 or i == 1:
            continue
        else:
            for j in range(2, int(i/2)+1):
                if i % j == 0:
                    break
            else:
                prime_list.append(i)
    return np.array(prime_list)

@jit("int32[:](int32, float64, float64, float64, int32[:], int32)", nopython=True, fastmath=False)
def del_list(n=16, min_dist=1., max_dist=100., curve=1.0, prime_list=np.array([0]), sr=44100):
    
    """
    create a list of delay times, with a minimum and maximum distance.
    
    n: number of delay times
    curve: exponential value to multiply each delay time by
    """
    
    # divided by speed of sound multiplied by milliseconds.
    min_dist  /= 344
    min_dist *= 1000.
    
    max_dist  /= 344.
    max_dist *= 1000.
    
    l = np.power(np.linspace(0, 1, n), curve)
    l = (((min_dist + (l * (max_dist - min_dist))) / 1000) * sr) + 0.5
    
    current = prime_list[(np.abs(prime_list - l[0])).argmin()]
    
    for i in range(n):
        
        index = (np.abs(prime_list - l[i])).argmin()
        
        l[i] = prime_list[index]
        
        while (current >= l[i]):
            index += 1
            l[i] = prime_list[index]
        
        current = l[i]
    
    return l.astype(np.int32)

@jit("float64[:](int32, float64, float64, float64)", nopython=True, fastmath=False)
def filter_list(n=16, min_freq=200., max_freq=1200., frequency_curve=3.5):
    l = np.power(np.linspace(0, 1, n), frequency_curve)
    l = min_freq + (l * (max_freq - min_freq))
    return l

@vectorize([float64(float64)])
def t60(duration_in_samples):
    return np.exp(np.log(0.001) / duration_in_samples)

@vectorize([float64(float64)])
def t60_time(multiplier):
    return np.log(0.001) / np.log(multiplier)

@jit("float64[:](float64[:], float64, float64, float64, float64, float64, float64, float64, float64[:, :], int32[:], int64, int64)", nopython=True, fastmath=True, parallel=False)
def simple_fdn(input,
               decay=None,
               min_dist=1.,
               max_dist=100.,
               distance_curve=3.,
               min_freq=11000.,
               max_freq=18000.,
               frequency_curve=1.809,
               H=np.array(0),
               prime_list=np.array(0),
               sr=44100,
               max_length=2000):

    # assert all values have been entered.
    # assert decay != None, f"decay between 0 and 1 expected, got: {decay}"
    # assert np.abs(decay) <= 1.0, f"decay between 0 and 1 expected, got: {decay}"

    # scale delay time parameter values if they are within range.
    min_dist = min_dist * 100. if (min_dist > 0) else 1.
    max_dist = max_dist * 100. if (max_dist > 0) else 1.
    
    # swap if minimum is larger than maximum
    if (min_dist > max_dist):
        min_dist, max_dist = max_dist, min_dist
    
    distance_curve = (distance_curve + 1.0) * 3. if (distance_curve > 0.) else 1.5
    
    min_freq = min_freq * 20000 if (min_freq > 0) else 20000
    max_freq = max_freq * 20000 if (max_freq > 0) else 20000
    
    if (min_freq > max_freq):
        min_freq, max_freq = max_freq, min_freq
    
    frequency_curve = (frequency_curve + 1.0) * 3. if (frequency_curve > 0.) else 1.5
    
    l = del_list(n=16, min_dist=min_dist, max_dist=max_dist, curve=distance_curve, prime_list=prime_list, sr=sr)
    
    N = l.shape[0]
    
    freqs = filter_list(n=N, min_freq=min_freq, max_freq=max_freq, frequency_curve=frequency_curve)

    # default smallest decay time equivalent to largest delay length in milliseconds
    decay = decay * max_length if (decay > 0.) else max_dist
    
    # empty array to calculate each t60 value.
    decay_ms = np.zeros(N)
    
    # print gain calculation for each delay line as a function of t60
    # convert ms to samples and then divide by loop_time in samples then take t60.
    for i in range(N):
        decay_ms[i] = t60(((decay/1000) * sr) / l[i])
        
        
    # all generated impulse responses should be 2 seconds.
    # if processing a real world signal the sample may be larger than two seconds.
    # check for that case and leave if less than 88200 add until it is
    
    extra_time_in_samples = int(t60_time(decay_ms[-1]) * l[-1])
    difference = (sr * 2) - extra_time_in_samples
    if (difference) > 0: extra_time_in_samples += difference 
    
    x = np.zeros((N, input.shape[-1] + extra_time_in_samples))
    x[:, :input.shape[-1]] = input

    y = np.zeros(x.shape)
    f_y = np.zeros(x.shape)
    output = np.zeros(x.shape)
    feedback = np.zeros(x.shape)
    feedback[:, :] = x[:, :]
    
    num_iterations = int(np.ceil((44100 / l[0])  * (decay / 1000)))
    
    for j in prange(num_iterations):
        for i in prange(N):
            y[i] = feedforward_delay(feedback[i] * decay_ms[i], l[i], milliseconds=False, sr=sr)
            f_y[i] = onepole(y[i], freq=freqs[i], sr=sr)
            output[i] += dc_block(f_y[i], sr=sr)
        feedback = H @ f_y
        
    y = np.sum(output, axis=0)
    
    return y * (1 / np.max(np.abs(y)))