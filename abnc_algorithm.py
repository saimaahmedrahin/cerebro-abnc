"""
ABNC: Active Brain Noise Cancellation (prototype)
Cerebro Solutions - Saima Ahmed Rahin
License: MIT. Research use only, not a medical device.

Pipeline: 50/60 Hz notch -> zero-phase 0.8-8 Hz bandpass.
Measured on 400 simulated 64 s recordings (see evaluate()):
  SNR -3.1 dB raw -> 16.9 dB after ABNC (+20 dB);
  +6.8 dB over a standard 0.5-40 Hz clinical filter;
  <1% error in 1-4 Hz delta-band power; ~2 ms per 64 s channel.
"""

import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.fft import rfft, rfftfreq

FS = 250


def abnc_denoise(x, fs=FS, line_freq=60, band=(0.8, 8.0)):
    """Clean one EEG channel. x: 1D array. Returns array of the same length."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {x.shape}")
    b, a = iirnotch(line_freq, Q=30, fs=fs)
    x = filtfilt(b, a, x)
    sos = butter(4, list(band), btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def abnc_denoise_multichannel(eeg, fs=FS, **kwargs):
    """eeg: array of shape (channels, samples)."""
    eeg = np.asarray(eeg, dtype=float)
    return np.vstack([abnc_denoise(ch, fs=fs, **kwargs) for ch in eeg])


def snr_db(clean, test):
    return 10 * np.log10(np.mean(clean ** 2) / np.mean((test - clean) ** 2))


def delta_power_error_pct(clean, test, fs=FS, lo=1.0, hi=4.0):
    f = rfftfreq(clean.size, 1 / fs)
    m = (f >= lo) & (f <= hi)
    pc = np.sum(np.abs(rfft(clean))[m] ** 2)
    pt = np.sum(np.abs(rfft(test))[m] ** 2)
    return 100 * abs(pt - pc) / pc


def simulate_pair(rng, seconds=64, fs=FS):
    """Simulated clean EEG and a noisy copy (drift + powerline + sensor noise)."""
    t = np.arange(seconds * fs) / fs
    clean = (2.0 * np.sin(2 * np.pi * 1.5 * t + rng.uniform(0, 2 * np.pi))
             + 1.5 * np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 2 * np.pi))
             + 0.5 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi)))
    noise = (3.0 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
             + 1.5 * np.sin(2 * np.pi * 60 * t)
             + rng.normal(0, 1.0, t.size))
    return clean, clean + noise


def baseline_filter(x, fs=FS):
    sos = butter(4, [0.5, 40], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def evaluate(n=400, seed=42):
    rng = np.random.default_rng(seed)
    raw, base, abnc, perr, ms = [], [], [], [], []
    for _ in range(n):
        c, y = simulate_pair(rng)
        t0 = time.perf_counter()
        z = abnc_denoise(y)
        ms.append((time.perf_counter() - t0) * 1000)
        raw.append(snr_db(c, y))
        base.append(snr_db(c, baseline_filter(y)))
        abnc.append(snr_db(c, z))
        perr.append(delta_power_error_pct(c, z))
    print(f"Signals tested:            {n} (simulated, not patient data)")
    print(f"SNR raw:                   {np.mean(raw):6.2f} dB")
    print(f"SNR standard 0.5-40 Hz:    {np.mean(base):6.2f} dB")
    print(f"SNR ABNC:                  {np.mean(abnc):6.2f} dB")
    print(f"Gain vs raw:               {np.mean(abnc) - np.mean(raw):+6.2f} dB")
    print(f"Gain vs standard filter:   {np.mean(abnc) - np.mean(base):+6.2f} dB")
    print(f"Delta-band power error:    {np.mean(perr):6.2f} %")
    print(f"Time per 64 s channel:     {np.median(ms):6.2f} ms")


if __name__ == "__main__":
    evaluate()
