"""
Utility functions: STFT/iSTFT wrappers, audio I/O, and evaluation metrics.
"""

import torch
import numpy as np
import soundfile as sf


# ─── STFT configuration ───────────────────────────────────────────────────────

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512


def stft(x: torch.Tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    Compute STFT and return (B, F, T, 2) real-valued tensor.
    x: (B, L) waveform
    Returns: (B, F, T, 2) where F = n_fft//2 + 1
    """
    window = torch.hann_window(win_length, device=x.device)
    X = torch.stft(x, n_fft, hop_length, win_length, window=window, return_complex=False)
    # X shape: (B, F, T, 2)
    return X


def istft(X: torch.Tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    Inverse STFT from (B, F, T, 2) real-valued tensor.
    Returns: (B, L) waveform
    """
    window = torch.hann_window(win_length, device=X.device)
    # Convert to complex
    X_complex = torch.complex(X[..., 0], X[..., 1])
    x = torch.istft(X_complex, n_fft, hop_length, win_length, window=window)
    return x


# ─── Audio I/O ────────────────────────────────────────────────────────────────

def load_wav(path, sr=SAMPLE_RATE):
    """Load wav file as float32 numpy array, resampled to sr."""
    audio, orig_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    if orig_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio


def save_wav(path, audio, sr=SAMPLE_RATE):
    """Save numpy array as wav."""
    sf.write(path, audio, sr)


# ─── Evaluation metrics ───────────────────────────────────────────────────────

def si_snr(estimate, reference):
    """
    Scale-invariant SNR (in dB).
    estimate, reference: (B, L) or (L,)
    """
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
        reference = reference.unsqueeze(0)
    # zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    dot = torch.sum(estimate * reference, dim=-1, keepdim=True)
    s_ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + 1e-8
    proj = dot * reference / s_ref_energy

    noise = estimate - proj
    si_snr_val = 10 * torch.log10(
        torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8)
    )
    return si_snr_val  # (B,)


def snr(estimate, reference):
    """
    Signal-to-noise ratio (dB).
    """
    noise = estimate - reference
    return 10 * torch.log10(
        torch.sum(reference ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8)
    )
