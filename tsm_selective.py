import numpy as np
import librosa
from matplotlib import pyplot as plt
from IPython.display import display, Audio
from glob import glob
import os

from utils import draw_spec, audioshow
from stft import stft, extract_frames

# from tsm import SynchronousOLA, LSEE_TSM

""" 
Modified modules for 'frame selective' STFT and OLA
Enables frame selective TSM, which makes certain frame faster or slower

** overlapadd()
** istft()
** custom_stft()
** SynchronousOLA()
** modLSEE_TSM()

"""
def overlapadd(frames, win_length, Sa, frame_ranges, rate_ranges, win_type='hann', griffin=True):
    """
    Modified OLA proceSa with variable rates for specific frame ranges.
    input: 
        frames: frame list with [NumFrames]
        Sa: Base hop length
        frame_ranges: List of tuples [(start1, end1), (start2, end2), ...] for frames
        rate_ranges: List of rates [slow_rate, fast_rate, ...] corresponding to frame_ranges
        win_type: Window array or string type of window
    output:
        y: Reconstructed waveform using modified OLA.
    """
    if len(frame_ranges) != len(rate_ranges):
        raise ValueError("frame_ranges and rate_ranges must have the same length.")
    
    num_frames = len(frames)
    # print(frame_ranges)

    # import pdb
    # pdb.set_trace()
    # siglen = win_length + sum(
    #     int(Sa / rate_ranges[i]) if any(start <= idx <= end for i, (start, end) in enumerate(frame_ranges)) else Sa
    #     for idx in range(num_frames)
    # )

    siglen = win_length
    for idx in range(num_frames):
        # Determine rate for current frame
        hop_length = Sa
        for i, (start, end) in enumerate(frame_ranges):
            if start <= idx <= end:
                hop_length = int(Sa / rate_ranges[i])
                break
        siglen += hop_length

    y = np.zeros(siglen)
    window_sum = np.zeros(siglen)
    
    # Generate window
    if isinstance(win_type, str):
        try:
            window = librosa.filters.get_window(win_type, win_length)
        except ValueError:
            raise ValueError("Unsupported window type!")
    else:
        window = win_type  # ASaume win_type is already an array with proper padding
    
    current_position = 0  # Track the cumulative start position
    for frame_idx in range(num_frames):
        # Determine rate and hop length based on ranges
        hop_length = Sa
        for i, (start, end) in enumerate(frame_ranges):
            if start <= frame_idx <= end:
                hop_length = int(Sa / rate_ranges[i])
                break

        start = current_position
        frame = frames[frame_idx]
        
        if griffin:
            y[start:start + win_length] += frame * window
            window_sum[start:start + win_length] += window ** 2
        else:
            y[start:start + win_length] += frame
            window_sum[start:start + win_length] += window            
        # Update the current position based on the hop length
        current_position += hop_length
    
    # Normalize by window overlap factor
    y /= np.where(window_sum > 1e-10, window_sum, 1e-10)

    # Crop to remove paddings (center-based STFT)
    y = y[win_length // 2:-win_length // 2]
    window_sum = window_sum[win_length // 2:-win_length // 2]
    return y

def istft(Y_w, win_length, Sa, n_fft, frame_ranges, rate_ranges, win_type='hann', griffin=True):
    """
    Modified ISTFT to handle variable rates for specific frame ranges.
    ** Parameters:
        Y_w: STFT (complex) matrix
        win_length: Window length
        Sa: Base hop length
        frame_ranges: List of tuples [(start1, end1), (start2, end2), ...] for frames
        rate_ranges: List of rates [slow_rate, fast_rate, ...] corresponding to frame_ranges
        win_type: Window type
        griffin: Whether to use Griffin-Lim algorithm normalization
    ** Returns:
        y_buffer: Reconstructed time-domain signal
    """
    if not n_fft:
        n_fft = win_length  # Default to win_length if n_fft is not provided
        
    if win_length < Sa:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to Sa ({Sa})")
    if n_fft < win_length:
        raise ValueError(f"n_fft ({n_fft}) must be greater than or equal to win_length ({win_length})")

    # Generate window with padding if win_length < n_fft
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")
    padlen = n_fft - win_length
    padded_window = np.pad(window, (padlen // 2, padlen // 2), mode='constant')

    # Reconstruct Y to get full spectrum in frequency axis
    Y_flip = np.flipud(Y_w)[1:-1]
    Y_w = np.concatenate((Y_w, np.conj(Y_flip)), axis=0)  # Note that phase is odd
    
    num_frames = Y_w.shape[1]
    frames = []

    # Calculate each frame by inverse FFT
    for frame_idx in range(num_frames):
        frame = np.real(np.fft.ifft(Y_w[:, frame_idx]))  # Inverse FFT
        frames.append(frame)
    
    # Modified OLA with variable rates
    y_buffer = overlapadd(frames, win_length=win_length, Sa=Sa, frame_ranges=frame_ranges, 
                          rate_ranges=rate_ranges, win_type=padded_window, griffin=griffin)
    return y_buffer

def custom_stft(y, sr=16000, win_type='hamming', win_length=320, Sa=160, 
                frame_ranges=[(10, 20), (50, 75)], rate_ranges=[0.8, 1.2], 
                n_fft=None, pad_mode='constant'):
    """
    Compute STFT with variable rates for multiple frame ranges.
    ** Parameters:
        y: Input signal
        Sa: Base hop length
        frame_ranges: List of tuples [(start1, end1), (start2, end2), ...] for frames
        rate_ranges: List of rates [slow_rate, fast_rate, ...] corresponding to frame_ranges
    ** Returns:
        spec: Complex STFT (NFFT//2+1 x Frames)
    """
    if len(frame_ranges) != len(rate_ranges):
        raise ValueError("frame_ranges and rate_ranges must have the same length.")

    if not n_fft:
        n_fft = win_length
    
    if n_fft < win_length:
        raise ValueError(f"n_fft ({n_fft}) must be greater than or equal to win_length ({win_length})")

    y = np.pad(y, (n_fft // 2, n_fft // 2), mode=pad_mode, constant_values=0)  # Padding
    siglen_pad = len(y)  # Length of the padded signal

    # Generate window
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")

    # Initialize variables
    spec = []
    centers = []
    current_center = n_fft // 2  # Start from the center
    frame_idx = 0

    # Calculate centers based on variable rates
    while current_center + n_fft // 2 < siglen_pad:
        centers.append(current_center)

        # Determine rate and hop length based on ranges
        hop_length = Sa
        for i, (start, end) in enumerate(frame_ranges):
            if start <= frame_idx <= end:
                hop_length = int(Sa / rate_ranges[i])
                break

        current_center += hop_length
        frame_idx += 1

    # Process frames
    for center in centers:
        start = center - win_length // 2
        end = center + win_length // 2

        if end > siglen_pad:  # End condition
            break

        frame = y[start:end]
        frame = frame * window

        # Zero-pad to match n_fft
        padlen = n_fft - len(frame)
        frame = np.pad(frame, pad_width=[padlen // 2, padlen // 2], mode='constant')

        # Compute FFT and append to the spectrogram
        frame_fft = np.fft.fft(frame)[:n_fft // 2 + 1]
        spec.append(frame_fft)

    spec = np.array(spec).T  # [freq x timeframe]
    return spec

def SynchronousOLA(frames, win_length, Sa, frame_ranges, rate_ranges, win_type='hann', max_shift=130, visualize=True):
    """
    Modified SOLA (Synchronous Overlap-Add) for multiple frame ranges with variable rates.
    ** Inputs:
        frames: List of frames to be combined.
        win_length: Window length for each frame.
        Sa: Base hop length
        frame_ranges: List of tuples [(start1, end1), (start2, end2), ...] for frames
        rate_ranges: List of rates [slow_rate, fast_rate, ...] corresponding to frame_ranges
        win_type: Window type (default 'hann') or a window array.
        max_shift: Maximum allowed shift for alignment (in samples).
        visualize: Enable/disable visualization.
    ** Returns:
        y: Reconstructed signal using modified SOLA.
    """
    if len(frame_ranges) != len(rate_ranges):
        raise ValueError("frame_ranges and rate_ranges must have the same length.")

    num_frames = len(frames)
    # siglen = win_length + sum(
    #     int(Sa / rate_ranges[i]) if any(start <= idx <= end for i, (start, end) in enumerate(frame_ranges)) else Sa
    #     for idx in range(num_frames)
    # )
    siglen = win_length
    for idx in range(num_frames):
        # Determine rate for current frame
        hop_length = Sa
        for i, (start, end) in enumerate(frame_ranges):
            if start <= idx <= end:
                hop_length = int(Sa / rate_ranges[i])
                break
        siglen += hop_length

    y = np.zeros(siglen)
    window_sum = np.zeros(siglen)

    # Generate window
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")

    # Overlap Add with variable hop lengths
    current_position = 0  # Track the cumulative start position
    for frame_idx in range(num_frames):
        # Determine rate and hop length based on ranges
        hop_length = Sa
        for i, (start, end) in enumerate(frame_ranges):
            if start <= frame_idx <= end:
                hop_length = int(Sa / rate_ranges[i])
                break

        start = current_position
        frame = frames[frame_idx]

        if frame_idx == 0:
            # First frame is added normally
            y[start:start + win_length] += frame * window
            window_sum[start:start + win_length] += window ** 2
        else:
            overlap_start = start
            overlap_end = start + win_length
            existing_segment = y[overlap_start:overlap_end]

            # Calculate cross-correlation and find the best shift
            correlations = np.correlate(existing_segment, frame, mode='full')
            max_idx = np.argmax(correlations[len(correlations)//2 - max_shift:len(correlations)//2 + max_shift + 1])
            best_shift = max_idx - max_shift

            # Apply best shift and combine the frame
            shifted_frame = np.roll(frame, best_shift)
            y[overlap_start:overlap_end] += shifted_frame * window
            window_sum[overlap_start:overlap_end] += window ** 2

        # Update the current position
        current_position += hop_length

    # Normalize
    y /= np.where(window_sum > 1e-10, window_sum, 1e-10)

    # Crop out to remove paddings (if center-based STFT was used)
    y = y[win_length // 2:-win_length // 2]
    return y

def mod_LSEE_TSM(y, rate=2, win_length=512, hop_length=256, frame_ranges=[(10, 20), (75, 100)], 
                 rate_ranges=[0.5, 1.2], num_iterations=50, initial='sola', 
                 win_type='hann', verbose=True, griffin=True, return_ser=False):
    """
    Time-Scale Modification using LSEE-MSTFTM for phase reconstruction with variable rates.
    ** Parameters:
        y: Input signal
        rate: Default time-scaling rate (applied outside of frame_ranges).
        frame_ranges: List of frame ranges where rate_ranges are applied.
        rate_ranges: List of rates corresponding to frame_ranges.
        num_iterations: Number of iterations for phase reconstruction.
        initial: ['gaussian', 'sola', 'zero_phase'].
        return_ser: If True, returns SER list along with reconstructed signal.
    """
    from fractions import Fraction
    if len(frame_ranges) != len(rate_ranges):
        raise ValueError("frame_ranges and rate_ranges must have the same length.")

    n_fft = win_length
    Sa = hop_length  # Default hop length

    # Adjust length
    frac_rate = Fraction(rate).limit_denominator()
    y_len = (len(y) // frac_rate.numerator) * frac_rate.numerator
    y = y[:y_len]

    # STFTM of original signal
    # Y = np.abs(custom_stft(y, win_type=win_type, win_length=win_length, Ss=Ss, 
                        #    frame_ranges=frame_ranges, rate_ranges=rate_ranges))
    Y = np.abs(stft(y, n_fft=n_fft, win_type=win_type, win_length=win_length, hop_length=Sa, plot=False))
    if verbose:
        print(f"STFT of original signal (shape): {Y.shape}")

    # Initialize x_initial
    if initial == 'gaussian':
        x_initial = istft(Y, win_length=win_length, Sa=Sa, frame_ranges=frame_ranges, n_fft=n_fft,
                          rate_ranges=rate_ranges, win_type=win_type, griffin=griffin)
        x_initial = np.random.normal(0, 1, len(x_initial))  # Gaussian noise
    elif initial == 'sola':
        frames = extract_frames(y, win_type=win_type, win_length=win_length, hop_length=Sa)
        x_initial = SynchronousOLA(frames, win_length=win_length, Sa=Sa, frame_ranges=frame_ranges, 
                                   rate_ranges=rate_ranges, win_type=win_type, visualize=verbose)
    elif initial == 'zero_phase':
        X = Y
    else:
        raise ValueError(f"Unsupported initial method: {initial}")

    # Iterative phase reconstruction
    ser_list = []
    for i in range(num_iterations):
        if initial != 'zero_phase' or i > 0:
        # Compute STFT of current signal
            X = custom_stft(x_initial, win_type=win_type, win_length=win_length, Sa=Sa,
                            frame_ranges=frame_ranges, rate_ranges=rate_ranges)

        # Replace magnitude with original
        if X.shape[1] == Y.shape[1]:
            X_phase = np.angle(X)
            X_new = Y * np.exp(1j * X_phase)
        else:
            raise ValueError(f"Mismatch in STFT frame count: {X.shape[1]} != {Y.shape[1]}")

        # ISTFT to reconstruct signal
        x_initial = istft(X_new, win_length=win_length, Sa=Sa, frame_ranges=frame_ranges, n_fft=n_fft,
                          rate_ranges=rate_ranges, win_type=win_type, griffin=griffin)

    return x_initial

def main():
    dir_path = "./*.wav"
    # Example parameters
    WINLEN = 1024
    HOPLEN = 256  # Default hop length
    RATE = 0.5
    FRAME_RANGES = [(10, 35), (85, 120)]  # Multiple frame ranges
    RATE_RANGES = [0.8, 1.2]  # Rates for frame ranges
    NUM_ITERATIONS = 50
    INITIAL_METHOD = 'zero_phase'
    VERBOSE = True

    # Load audio
    y, sr = librosa.load(glob(dir_path)[0], sr=None)

    # Perform TSM with variable rates
    y_reconstructed = mod_LSEE_TSM(
        y, rate=RATE, win_length=WINLEN, hop_length=HOPLEN, frame_ranges=FRAME_RANGES, 
        rate_ranges=RATE_RANGES, num_iterations=NUM_ITERATIONS, initial=INITIAL_METHOD, 
        win_type='hann', verbose=VERBOSE
    )

    # Display results
    print(f"Original signal length: {len(y)}, Reconstructed signal length: {len(y_reconstructed)}")
    audioshow(y)
    audioshow(y_reconstructed)
    display(Audio(y, rate=sr))
    display(Audio(y_reconstructed, rate=sr))

if __name__ == "__main__":
    main()

