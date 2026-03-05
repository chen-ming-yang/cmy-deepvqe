"""
Training configuration — single place to change all hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── data paths ─────────────────────────────────────────────────────────
    # AEC-Challenge root: nearend_speech/, farend_speech/
    aec_root: Optional[str] = None          # e.g. "datasets/aec_challenge/synthetic"

    # DNS-Challenge root: clean/
    dns_root: Optional[str] = None          # e.g. "datasets/dns_challenge"

    # Shared noise directory (*.wav)
    noise_dir: Optional[str] = "/home/cmy/cmy/3D-Speaker/egs/3dspeaker/sv-eres2netv2/data/raw_data/musan"         # e.g. "datasets/noise"

    # Shared RIR directory (*.wav, optional — synthetic RIR used if None)
    rir_dir: Optional[str] = "/home/cmy/cmy/AEC-Challenge/datasets/RIRs"           # e.g. "datasets/rir"

    # Validation (same structure, or None to skip)
    val_aec_root: Optional[str] = None
    val_dns_root: Optional[str] = None
    val_noise_dir: Optional[str] = None
    val_rir_dir: Optional[str] = None

    # ── audio / STFT ──────────────────────────────────────────────────────
    sr: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    segment_len: float = 4.0               # seconds per training clip

    # ── mixing parameters ─────────────────────────────────────────────────
    snr_range: tuple = (-5, 20)             # near-end to noise SNR (dB)
    ser_range: tuple = (-10, 10)            # signal-to-echo ratio (dB)
    gain_range_db: tuple = (-6, 6)          # random gain (dB)
    rt60_range: tuple = (0.1, 0.7)          # synthetic RIR RT60
    echo_delay_range_ms: tuple = (1, 200)   # echo path delay
    echo_rt60_range: tuple = (0.05, 0.3)    # echo path RT60
    reverb_prob: float = 0.5                # reverb probability
    echo_prob: float = 1.0                  # echo probability (if far-end)
    noise_prob: float = 1.0                 # noise probability
    use_pregenerated_echo: bool = True      # use AEC echo_signal/ directly

    # ── model ─────────────────────────────────────────────────────────────
    mic_channels: Optional[List[int]] = None
    ref_channels: Optional[List[int]] = None
    dec_channels: Optional[List[int]] = None
    gru_hidden: int = 256
    align_hidden: int = 16
    dmax: int = 100
    fe_compress: float = 0.3

    # ── loss ──────────────────────────────────────────────────────────────
    compress: float = 0.3
    loss_alpha: float = 0.7                 # complex vs magnitude balance
    lambda_spec: float = 1.0
    lambda_sisnr: float = 0.1

    # ── training ──────────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"           # "cosine" | "step"
    lr_step_size: int = 30
    lr_gamma: float = 0.5
    grad_clip: float = 5.0
    num_workers: int = 4

    # ── checkpointing / logging ───────────────────────────────────────────
    save_dir: str = "checkpoints"
    log_interval: int = 50                  # print every N steps
    save_interval: int = 1                  # save every N epochs
    resume: Optional[str] = None            # path to checkpoint to resume
    seed: int = 42

    # ── device ────────────────────────────────────────────────────────────
    device: str = "cuda"                    # "cuda" | "cpu"
