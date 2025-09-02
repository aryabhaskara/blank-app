import numpy as np
from scipy.fft import fft
from scipy.stats import kurtosis, skew

def compute_fft(signal):
    n = len(signal)
    return np.abs(fft(signal))[:n // 2]

def plot_fft(signal, time):
    n = len(signal)
    dt = np.mean(np.diff(time))   # average step in seconds
    fs = 1.0 / dt                 # sampling frequency (Hz)
    freqs = np.fft.fftfreq(n, d=dt)[:n // 2]
    fft_vals = np.abs(fft(signal))[:n // 2]
    return freqs, fft_vals, fs

def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)

def extract_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'shape_factor': rms / (np.mean(np.abs(signal)) + 1e-10),
        'rms': rms,
        'impulse_factor': np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10),
        'peak_to_peak': np.ptp(signal),
        'kurtosis': kurtosis(signal),
        'crest_factor': np.max(np.abs(signal)) / (rms + 1e-10),
        'skewness': skew(signal),
    }

ordered_columns = [
    f'{f}_{a}' for f in
    ["mean", "std", "shape_factor", "rms", "impulse_factor","peak_to_peak","kurtosis","crest_factor","skewness"]
    for a in ['x','y','z']
]
