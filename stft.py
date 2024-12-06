import numpy as np
import librosa
from IPython.display import Audio, display
from matplotlib import pyplot as plt

""" 
Function list
** extract_frames()
** stft()
** overlapadd()
** istft()
** LSEE_MSTFTM()
** SER()
Refer to STFT, OLA(Griffin Lim's Algorithm), ISTFT 
"""

def extract_frames(y, win_type='hamming', win_length=320, hop_length=160,):
    """ 
    Extract frames identical to librosa STFT 
    ** Returns:
        frame list that contations every time-domain frames
    """
    if win_length < hop_length:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to hop_length ({hop_length})")
    
    y = np.pad(y, (win_length//2, win_length//2), mode='constant', constant_values=0)  # padding
    siglen_pad = len(y)  # Length of the padded signal

    # window
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")

    frame_list = []
    # Frame processing
    for center in range(win_length//2, siglen_pad, hop_length):
        if center > siglen_pad - win_length//2:
            break #end condition
        start = center - win_length//2
        end = center + win_length//2
        frame = y[start:end]
        frame = frame * window
        frame_list.append(frame)
    return frame_list

def stft(y, sr=16000, win_type='hamming', win_length=320, hop_length=160, n_fft=None,
         pad_mode='constant', figsize=(14, 4), cmap='viridis', 
         vmin=-50, vmax=40,
         use_colorbar=True, plot=False, return_fig=False):
    
    """ 
    STFT Implementation identical to librosa.stft 
    This implementation is based on center=='True' option
    ** Returns:
        spec: Magnitude spectrogram (NFFT//2+1 x Frames).
        Returns the figure if `return_fig` is True.
    """
    if not n_fft:
        n_fft = win_length
    
    if win_length < hop_length:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to hop_length ({hop_length})")
    if n_fft < win_length:
        raise ValueError(f"n_fft ({n_fft}) must be greater than or equal to win_length ({win_length})")
    
    siglen_sec = len(y)/sr
    y = np.pad(y, (n_fft//2, n_fft//2), mode=pad_mode, constant_values=0)  # padding
    siglen_pad = len(y)  # Length of the padded signal

    # window
    try:
        window = librosa.filters.get_window(win_type, win_length)
    except ValueError:
        raise ValueError("Unsupported window type!")

    spec = []
    # Frame processing
    for center in range(n_fft//2, siglen_pad, hop_length):
        if center > siglen_pad - n_fft//2:
            break #end condition

        start = center - win_length//2
        end = center + win_length//2
        frame = y[start:end]
        frame = frame * window

        # pad until n_fft       
        padlen = n_fft - len(frame)
        frame = np.pad(frame, pad_width=[padlen//2, padlen//2], mode='constant')
        frame_fft = np.fft.fft(frame)[:n_fft//2 + 1]
        spec.append(frame_fft)

    spec = np.array(spec).T  # [freq x timeframe]
    # spec = np.abs(spec)

    # Plot option
    if plot:
        fig = plt.figure(figsize=figsize)
        # spec = 20 * np.log10(np.abs(spec))
        plt.imshow(20 * np.log10(np.abs(spec)), aspect='auto', 

                   cmap=cmap, 
                   vmin=vmin, vmax=vmax,
                   origin='lower', extent=[0, siglen_sec, 0, sr//2])

        if use_colorbar: plt.colorbar()
        plt.title("STFT Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

        if return_fig:
            plt.close()
            return spec, fig
        else:
            plt.show()
            return spec
    else:
        return spec

def overlapadd(frames, win_length, hop_length, win_type='hann', griffin=True):
    """ 
    OLA process and implementation of LSEE-MSTFT (Griffin Lim's)
    input: 
        frames: frame list with [NumFrames]
        win_type: Window array or string type of window
    output:
        y: Reconstructed waveform using OLA.
    """
    num_frames = len(frames)
    siglen = win_length + (num_frames - 1) * hop_length
    y = np.zeros(siglen)
    window_sum = np.zeros(siglen)
    
    # Generate window if win_type is a string; otherwise, use the provided array
    if isinstance(win_type, str):
        try:
            window = librosa.filters.get_window(win_type, win_length)
        except ValueError:
            raise ValueError("Unsupported window type!")
    else:
        window = win_type  # Assume win_type is already an array with proper padding
    
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        frame = frames[frame_idx]
        if griffin:
            y[start:start + win_length] += frame * window
            window_sum[start:start + win_length] += window ** 2
        else:
            y[start:start+win_length] += frame
            window_sum[start:start+win_length] += window
    
    # Normalize by window overlap factor
    y /= np.where(window_sum > 1e-10, window_sum, 1e-10)

    # crop out to remove paddings (center-based STFT)
    y = y[win_length//2:-win_length//2]
    window_sum = window_sum[win_length//2:-win_length//2]
    return y

def istft(Y_w, win_length, hop_length, n_fft, win_type='hann', griffin=True):
    """
    ISTFT Implementation identical to librosa.istft     
    ** Returns:
        y_buffer: Reconstructed time-domain signal
    """
    if not n_fft:
        n_fft = win_length  # Default to win_length if n_fft is not provided
        
    if win_length < hop_length:
        raise ValueError(f"win_length ({win_length}) must be greater than or equal to hop_length ({hop_length})")
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
    
    # OLA
    y_buffer = overlapadd(frames, win_length=n_fft, hop_length=hop_length, win_type=padded_window, griffin=griffin)
    return y_buffer

def ola_tsm(signal, win_length, hop_in, hop_out, win_type='hann'):
    """
    Perform Time-Scale Modification (TSM) using simple Overlap-Add (OLA).
    
    Parameters:
        signal (np.ndarray): Input signal (time-domain).
        win_length (int): Window length (in samples).
        hop_in (int): Hop length for the input signal.
        hop_out (int): Hop length for the output signal (TSM factor).
        win_type (str): Type of window function ('hann', 'hamming', etc.).
    
    Returns:
        output_signal (np.ndarray): Time-domain signal after TSM.
    """
    frames = extract_frames(signal, win_length=win_length, hop_length=hop_in, win_type=win_type)
    output_signal = overlapadd(frames, win_length=win_length, hop_length=hop_out, win_type=win_type, griffin=True)

    return output_signal

def LSEE_MSTFTM(stftm, win_length, hop_length, n_fft, num_iterations=50, win_type='hann', griffin=True, verbose=True):
    """
    LSEE-MSTFTM: Least-Squares Error Estimate for Modified Short-Time Fourier Transform Magnitude
    Iteratively estimates phase spectra from magnitude spectra
    ** Inputs:
        stftm: given magnitude spectrogram (|Y|) 
        win_length: Window length
        hop_length: Hop length
        n_fft: FFT size
        num_iterations: Number of iterations for phase reconstruction
        win_type: Type of window (default 'hann')
    ** Returns:
        y_reconstructed: Reconstructed time-domain signal
    """
    
    # Step 1: Initialize 
    y_initial = istft(stftm, win_length=win_length, hop_length=hop_length, n_fft=n_fft, win_type=win_type, griffin=griffin)
    # y_initial = np.random.normal(0,1,y_initial.shape) # Gaussian initialization
    
    for i in range(num_iterations):
        # Step 2: Compute STFT of the current time-domain signal
        Y = librosa.stft(y_initial, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=win_type)

        # Step 3: Replace magnitude with given stftm and retain phase
        Y_phase = np.angle(Y)
        Y_new = stftm * np.exp(1j * Y_phase)
        
        # SER
        current_magnitude = np.abs(Y)
        ser_value = SER(stftm, current_magnitude)
        if verbose:
            print(f"Iteration {i}/{num_iterations}, SER: {ser_value:.2f} dB")
        
        # Step 4: Perform ISTFT to get the updated time-domain signal
        y_initial = istft(Y_new, win_length=win_length, hop_length=hop_length, n_fft=n_fft, win_type=win_type, griffin=griffin)
        
    # Final SER
    Y_final = librosa.stft(y_initial, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=win_type)
    final_magnitude = np.abs(Y_final)
    final_ser_value = SER(stftm, final_magnitude)
    print(f"Final SER after {num_iterations} iterations: {final_ser_value:.2f} dB")

    return y_initial

def SER(original_magnitude, reconstructed_magnitude, epsilon=0):
    """
    Calculate Spectral Error Ratio (SER) in dB.
    
    Parameters:
        original_magnitude (np.ndarray): Original magnitude spectrogram (|Y_w|)
        reconstructed_magnitude (np.ndarray): Reconstructed magnitude spectrogram (|X_w|)
    
    Returns:
        float: SER value in dB
    """
    # Calculate the energy of the original magnitude in the numerator
    original_energy = np.sum(original_magnitude ** 2)
    
    # Calculate the squared error in the denominator
    error = np.sum((original_magnitude - reconstructed_magnitude) ** 2)
    
    if error ==0:
        return -250 # -infty(-250dB) for zero error
    
    # Compute SER in dB
    ser_value = 10 * np.log10(original_energy / (error+epsilon))
    
    return ser_value