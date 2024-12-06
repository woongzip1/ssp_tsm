from stft import extract_frames, stft, istft, SER
import numpy as np
import librosa
from matplotlib import pyplot as plt

"""
Function list
** SynchronousOLA(): SOLA that replaces OLA
** LSEE_TSM(): LSEE-MSTFTM based time scale modification
"""

def SynchronousOLA(frames, win_length, hop_length, win_type='hann', max_shift=130, visualize=True):
    """
    SOLA (Synchronous Overlap-Add) function for frame-based synthesis.
    ** Inputs:
        frames: List of frames to be combined.
        win_length: Window length for each frame.
        hop_length: Hop length for original signal synthesis.
        win_type: Window type (default 'hann') or a window array.
        max_shift: Maximum allowed shift for alignment (in samples).
    ** Returns:
        y: Reconstructed signal using SOLA.
    """
    num_frames = len(frames)
    siglen = win_length + (num_frames - 1) * hop_length
    y = np.zeros(siglen)
    window_sum = np.zeros(siglen)

    # window
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")

    # visualization options
    if visualize:
        vis_range=(22,25)
        vis_a, vis_b = vis_range  
        plot_count = vis_b - vis_a + 1
        if plot_count > 0:
            fig, axes = plt.subplots(1, plot_count, figsize=(plot_count * 5, 5))
            if plot_count == 1:
                axes = [axes]

    # Overlap Add
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
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

            # visualization options
            if visualize and vis_a <= frame_idx <= vis_b:
                ax = axes[frame_idx - vis_a]
                ax.plot(existing_segment, label='Existing Segment')
                ax.plot(frame, label='Original Frame (No Shift)', linestyle='dotted')
                ax.plot(np.roll(frame, best_shift), label=f'Shifted Frame (Best Shift={best_shift})', linestyle='dashed')
                ax.set_title(f'Frame {frame_idx}: OLA')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Amplitude')
                ax.legend(loc='upper right')
                ax.grid(True)
                
            # Apply best shift and combine the frame
            shifted_frame = np.roll(frame, best_shift)
            y[overlap_start:overlap_end] += shifted_frame * window
            window_sum[overlap_start:overlap_end] += window ** 2

    # Normalize
    y /= np.where(window_sum > 1e-10, window_sum, 1e-10)

    # Crop out to remove paddings (if center-based STFT was used)
    y = y[win_length // 2:-win_length // 2]
    return y

def LSEE_TSM(y, rate=2, hop_length=256, win_length=512, win_type='hamming', num_iterations=50, 
             verbose=True, griffin=True, initial='gaussian', return_ser=False):
    from fractions import Fraction
    """
    Time-Scale Modification using LSEE-MSTFTM for phase reconstruction.
    ** Inputs:
        y: Original input signal
        rate: Time-scaling factor (e.g., rate=2 means 2x speed-up)
        hop_length: Hop length for the original signal STFT
        win_length: Window length
        win_type: Window type (default 'hamming')
        num_iterations: Number of iterations for phase reconstruction
        verbose: Print SER at each iteration
        initial: ['gaussian', 'sola', 'zero_phase']
    ** Returns:
        x_reconstructed: Time-scaled and phase-reconstructed signal
    """
    n_fft = win_length
    Sa = hop_length 
    Ss = int(np.round(Sa / rate))  

    # Length adjustment
    frac_rate = Fraction(rate).limit_denominator()
    n = frac_rate.numerator
    y_len = (len(y) // n) * n 
    y = y[:y_len]
    
    # STFTM of original
    Y = np.abs(stft(y, n_fft=n_fft, win_type=win_type, win_length=win_length, hop_length=Sa, plot=False))
    if verbose: print(f"STFT of original signal (shape): {Y.shape}") 
    
    # Initial estmiate of TSM signal
    x_len = int(len(y) / rate)
    if initial == 'gaussian':
        x_initial = np.random.normal(0, 1, x_len)  
    elif initial  == 'sola':
        ################################### Key implementations ###################################
        frames = extract_frames(y, win_type=win_type, win_length=win_length, hop_length=Sa)
        x_initial = SynchronousOLA(frames, win_length=win_length, hop_length=Ss, win_type=win_type, visualize=verbose)
        ################################### Key implementations ###################################
    elif initial == 'zero_phase':
        X = Y
    else: 
        raise ValueError(f"Unsupported initial method: {initial}")
    
    ser_list = []
    for i in range(num_iterations):
        # Compute STFT of the current time-domain signal x
        if initial != 'zero_phase' or i > 0:
            X = stft(x_initial, n_fft=n_fft, hop_length=Ss, win_length=win_length, win_type=win_type)

        # Replace magnitude into STFTM (Y)
        if X.shape[1] == Y.shape[1]:
            X_phase = np.angle(X)
            X_new = Y * np.exp(1j * X_phase)
        else:
            raise ValueError(f"Mismatch in STFT frame count. {X.shape[1]}<>{Y.shape[1]} Please adjust the rate or signal length.")

        # Calculate SER (Spectral Error Ratio)
        current_magnitude = np.abs(X)
        ser_value = SER(Y, current_magnitude)
        ser_list.append(ser_value)
        if verbose:
            print(f"Iteration {i}/{num_iterations}, SER: {ser_value:.2f} dB")

        # ISTFT (OLA) and update the estimate
        # from librosa import istft
        # x_initial = librosa.istft(X_new, hop_length=Ss, win_length=win_length, window=win_type)
        x_initial = istft(X_new, win_length=win_length, hop_length=Ss, n_fft=n_fft, win_type=win_type, griffin=griffin)
    final_ser_value = SER(Y, np.abs(librosa.stft(x_initial, n_fft=n_fft, hop_length=Ss, win_length=win_length, window=win_type)))
    print(f"Final SER after {num_iterations} iterations: {final_ser_value:.2f} dB")
    ser_list.append(final_ser_value)

    if return_ser:
        return x_initial, ser_list
    else:
        return x_initial