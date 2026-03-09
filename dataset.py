"""
Unified dataset for DeepVQE training.

Supports both AEC-Challenge and DNS-Challenge data through a single pipeline:

  1. Load random near-end speech and far-end signal
  2. Sample random parameters (SNR, SER, gain, reverb)
  3. Apply room reverberation to near-end speech (RIR file or synthetic)
  4. Create echo from far-end signal (convolve with echo path)
  5. Add background noise at random SNR
  6. Compose:  mic = reverb_nearend + echo + noise
  7. Apply random gain
  8. Normalize to prevent clipping

  Return:  mic_spec, ref_spec (far-end), clean_spec (dry near-end target)

Data layouts
────────────
AEC-Challenge synthetic (https://github.com/microsoft/AEC-Challenge):
    aec_root/
      ├── farend_speech/        # far-end signals (some with bg noise)
      ├── echo_signal/          # transformed far-end → echo
      ├── nearend_speech/       # clean near-end target
      ├── nearend_mic_signal/   # mic = nearend + echo (+ noise)
      └── meta.csv              # fileid, ser, nearend_scale, is_nearend_noisy, …

    Files are linked by name:  fileid_{int}.wav

DNS-Challenge:
    dns_root/
      └── clean/                # clean speech (target, no echo)
    or:
    dns_root/
      ├── datasets.clean.emotional_speech/
      ├── datasets.clean.french_data/
      └── ...                   # any subdirs with .wav files

Noise directory  (shared by both):
    noise_dir/
      └── *.wav                 # noise clips

RIR directory   (optional, shared by both):
    rir_dir/
      └── *.wav                 # room impulse responses
"""

import os
import glob
import random
import csv

import numpy as np
import torch
from scipy import signal as scipy_signal
from torch.utils.data import Dataset

from utils import load_wav, stft, SAMPLE_RATE


# ─── helpers ──────────────────────────────────────────────────────────────────

def _scan_wavs(folder):
    """Return sorted list of .wav paths under folder (recursive).
    
    Args:
        folder: can be a single path (str) or list of paths (List[str])
    """
    if folder is None:
        return []
    files = glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)
    files += glob.glob(os.path.join(folder, "**", "*.WAV"), recursive=True)
    files.sort()
    return files
    
    # Handle both single path and list of paths
    folders = [folder] if isinstance(folder, str) else folder
    
    all_files = []
    for f in folders:
        if f is not None:
            files = glob.glob(os.path.join(f, "**", "*.wav"), recursive=True)
            all_files.extend(files)
    
    all_files.sort()
    return all_files


def _rand_crop(audio, length):
    """Random crop / zero-pad to exactly *length* samples."""
    if len(audio) >= length:
        start = random.randint(0, len(audio) - length)
        return audio[start: start + length]
    else:
        pad = np.zeros(length, dtype=np.float32)
        pad[:len(audio)] = audio
        return pad


# ─── signal processing helpers ────────────────────────────────────────────────

def _scale_to_snr(signal, interference, snr_db):
    """Scale *interference* so that signal / interference = snr_db."""
    sig_power = np.mean(signal ** 2) + 1e-8
    int_power = np.mean(interference ** 2) + 1e-8
    scale = np.sqrt(sig_power / (int_power * 10 ** (snr_db / 10)))
    return interference * scale


def _generate_synthetic_rir(sr=16000, rt60_range=(0.1, 0.7)):
    """
    Generate a simple synthetic RIR using exponentially decaying noise.
    """
    rt60 = random.uniform(*rt60_range)
    length = max(int(rt60 * sr), int(0.05 * sr))
    t = np.arange(length, dtype=np.float32)
    decay = np.exp(-6.9078 * t / (rt60 * sr))       # -60 dB at rt60
    rir = np.random.randn(length).astype(np.float32) * decay
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir ** 2) + 1e-8)
    return rir


def _convolve(audio, filt):
    """FFT-based convolution, keeping original length and energy."""
    out = scipy_signal.fftconvolve(audio, filt, mode="full")[:len(audio)]
    orig_e = np.sqrt(np.mean(audio ** 2) + 1e-8)
    out_e  = np.sqrt(np.mean(out ** 2) + 1e-8)
    out = out * (orig_e / out_e)
    return out.astype(np.float32)


def _generate_echo_path(sr=16000, delay_range_ms=(1, 200), rt60_range=(0.05, 0.3)):
    """
    Generate an echo path impulse response:
    a delayed, decaying response that models acoustic coupling from
    loudspeaker → microphone.
    """
    delay_ms = random.uniform(*delay_range_ms)
    delay_samples = int(delay_ms * sr / 1000)
    rt60 = random.uniform(*rt60_range)
    tail_len = max(int(rt60 * sr), int(0.02 * sr))
    total_len = delay_samples + tail_len

    h = np.zeros(total_len, dtype=np.float32)
    t = np.arange(tail_len, dtype=np.float32)
    decay = np.exp(-6.9078 * t / (rt60 * sr))
    h[delay_samples:] = np.random.randn(tail_len).astype(np.float32) * decay
    h /= np.sqrt(np.sum(h ** 2) + 1e-8)
    return h


def _normalize(audio, peak_db=-1.0):
    """Peak-normalize to prevent clipping. Target peak in dBFS."""
    peak = np.max(np.abs(audio)) + 1e-8
    target = 10 ** (peak_db / 20)
    return (audio * target / peak).astype(np.float32)


# ─── Unified dataset ──────────────────────────────────────────────────────────

class SpeechEnhancementDataset(Dataset):
    """
    Unified on-the-fly mixing dataset.

    Each __getitem__ runs the 8-step pipeline:
      1  load random near-end + far-end
      2  sample random parameters
      3  apply RIR to near-end  →  reverb_nearend
      4  create echo from far-end
      5  add background noise
      6  mic = reverb_nearend + echo + noise
      7  random gain
      8  normalize

    Works for both AEC-Challenge (has far-end) and DNS (far-end = zeros).
    """

    def __init__(
        self,
        nearend_files,                       # list of clean near-end .wav paths
        farend_files=None,                   # list of far-end .wav paths (None → no echo)
        echo_files=None,                     # list of pre-generated echo .wav paths (optional)
        noise_files=None,                    # list of noise .wav paths
        rir_files=None,                      # list of RIR .wav paths (None → synthetic)
        sr=SAMPLE_RATE,
        segment_len=4.0,
        n_fft=512,
        hop=256,
        # ── random parameter ranges ──
        snr_range=(-5, 20),                  # near-end-to-noise SNR (dB)
        ser_range=(-10, 10),                 # signal-to-echo ratio (dB)
        gain_range_db=(-6, 6),               # random gain (dB)
        rt60_range=(0.1, 0.7),               # synthetic RIR RT60
        echo_delay_range_ms=(1, 200),        # echo path delay
        echo_rt60_range=(0.05, 0.3),         # echo path RT60
        reverb_prob=0.5,                     # probability of applying reverb
        echo_prob=1.0,                       # probability of adding echo (if far-end available)
        noise_prob=1.0,                      # probability of adding noise
        use_pregenerated_echo=False,         # if True, use echo_files instead of generating
    ):
        super().__init__()
        self.nearend_files = nearend_files
        self.farend_files = farend_files or []
        self.echo_files = echo_files or []
        self.noise_files = noise_files or []
        self.rir_files = rir_files or []
        self.sr = sr
        self.seg_samples = int(segment_len * sr)
        self.n_fft = n_fft
        self.hop = hop
        self.snr_range = snr_range
        self.ser_range = ser_range
        self.gain_range_db = gain_range_db
        self.rt60_range = rt60_range
        self.echo_delay_range_ms = echo_delay_range_ms
        self.echo_rt60_range = echo_rt60_range
        self.reverb_prob = reverb_prob
        self.echo_prob = echo_prob
        self.noise_prob = noise_prob
        self.use_pregenerated_echo = use_pregenerated_echo and len(self.echo_files) > 0

        print(f"[SpeechEnhancementDataset] "
              f"nearend={len(self.nearend_files)}, "
              f"farend={len(self.farend_files)}, "
              f"echo={len(self.echo_files)}, "
              f"noise={len(self.noise_files)}, "
              f"rir={len(self.rir_files)}, "
              f"use_pregenerated_echo={self.use_pregenerated_echo}")

    def __len__(self):
        return len(self.nearend_files)

    def __getitem__(self, idx):
        # ── 1. Load random near-end and far-end signals ──────────────────
        nearend = load_wav(self.nearend_files[idx], self.sr)
        nearend = _rand_crop(nearend, self.seg_samples)

        if self.farend_files:
            farend = load_wav(self.farend_files[idx], self.sr)  # strictly paired with nearend
            farend = _rand_crop(farend, self.seg_samples)
        else:
            farend = np.zeros(self.seg_samples, dtype=np.float32)

        # ── 2. Sample random parameters ──────────────────────────────────
        snr_db  = random.uniform(*self.snr_range)
        ser_db  = random.uniform(*self.ser_range)
        gain_db = random.uniform(*self.gain_range_db)
        do_reverb = random.random() < self.reverb_prob
        do_echo  = random.random() < self.echo_prob and len(self.farend_files) > 0
        do_noise = random.random() < self.noise_prob and len(self.noise_files) > 0

        # ── 3. Apply reverberation to near-end speech ────────────────────
        if do_reverb:
            if self.rir_files:
                # Use a real RIR file
                rir = load_wav(random.choice(self.rir_files), self.sr)
            else:
                # Generate synthetic RIR
                rir = _generate_synthetic_rir(self.sr, self.rt60_range)
            reverb_nearend = _convolve(nearend, rir)
        else:
            reverb_nearend = nearend.copy()

        # ── 4. Create echo from far-end signal ───────────────────────────
        if do_echo:
            if self.use_pregenerated_echo:
                # Use strictly paired pre-generated echo_signal file
                echo = load_wav(self.echo_files[idx % len(self.echo_files)], self.sr)
                echo = _rand_crop(echo, self.seg_samples)
                # Scale echo to target SER relative to near-end
                echo = _scale_to_snr(reverb_nearend, echo, ser_db)
            else:
                if self.rir_files and random.random() < 0.5:
                    echo_path = load_wav(random.choice(self.rir_files), self.sr)
                else:
                    echo_path = _generate_echo_path(
                        self.sr, self.echo_delay_range_ms, self.echo_rt60_range
                    )
                echo_raw = scipy_signal.fftconvolve(farend, echo_path, mode="full")[:self.seg_samples]
                echo_raw = echo_raw.astype(np.float32)
                # Scale echo to target SER relative to near-end
                echo = _scale_to_snr(reverb_nearend, echo_raw, ser_db)
        else:
            echo = np.zeros(self.seg_samples, dtype=np.float32)

        # ── 5. Add background noise ──────────────────────────────────────
        if do_noise:
            noise = load_wav(random.choice(self.noise_files), self.sr)
            noise = _rand_crop(noise, self.seg_samples)
            noise = _scale_to_snr(reverb_nearend, noise, snr_db)
        else:
            noise = np.zeros(self.seg_samples, dtype=np.float32)

        # ── 6. Compose mic signal ────────────────────────────────────────
        mic = reverb_nearend + echo + noise

        # ── 7. Apply random gain ─────────────────────────────────────────
        gain = 10 ** (gain_db / 20)
        mic = mic * gain
        farend = farend * gain            # ref sees same gain in practice

        # ── 8. Normalize to prevent clipping ─────────────────────────────
        mic    = _normalize(mic)
        farend = _normalize(farend)

        # ── Target is the dry clean near-end ─────────────────────────────
        clean = nearend

        # ── STFT ─────────────────────────────────────────────────────────
        mic_spec   = stft(torch.from_numpy(mic).unsqueeze(0),
                          self.n_fft, self.hop).squeeze(0)        # (F, T, 2)
        ref_spec   = stft(torch.from_numpy(farend).unsqueeze(0),
                          self.n_fft, self.hop).squeeze(0)
        clean_spec = stft(torch.from_numpy(clean).unsqueeze(0),
                          self.n_fft, self.hop).squeeze(0)

        return mic_spec, ref_spec, clean_spec


# ─── Convenience constructors ─────────────────────────────────────────────────

def _parse_aec_meta(aec_root):
    """
    Parse meta.csv from AEC-Challenge synthetic directory.
    Returns a dict:  fileid → {ser, nearend_scale, is_nearend_noisy, is_farend_noisy, …}
    """
    meta_path = os.path.join(aec_root, "meta.csv")
    meta = {}
    if not os.path.isfile(meta_path):
        return meta
    with open(meta_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row.get("fileid", row.get("file_id", "")).strip()
            if fid:
                meta[fid] = row
    return meta


def _build_aec_file_lists(aec_root):
    """
    Scan AEC-Challenge synthetic directory and return matched file lists.

    aec_root/
      ├── farend_speech/        fileid_{int}.wav
      ├── echo_signal/          fileid_{int}.wav
      ├── nearend_speech/       fileid_{int}.wav
      ├── nearend_mic_signal/   fileid_{int}.wav
      └── meta.csv

    Returns:
        nearend_files, farend_files, echo_files  (matched by fileid)
    """
    nearend_dir = os.path.join(aec_root, "nearend_speech")
    farend_dir  = os.path.join(aec_root, "farend_speech")
    echo_dir    = os.path.join(aec_root, "echo_signal")

    nearend_all = _scan_wavs(nearend_dir)
    farend_all  = _scan_wavs(farend_dir)
    echo_all    = _scan_wavs(echo_dir)

    # Build stem → path maps
    nearend_map = {os.path.splitext(os.path.basename(f))[0]: f for f in nearend_all}
    farend_map  = {os.path.splitext(os.path.basename(f))[0]: f for f in farend_all}
    echo_map    = {os.path.splitext(os.path.basename(f))[0]: f for f in echo_all}

    # Match by common fileids that appear in all three directories
    common = sorted(set(nearend_map) & set(farend_map) & set(echo_map))

    if not common:
        # Fallback: match nearend + farend only
        common_nf = sorted(set(nearend_map) & set(farend_map))
        if common_nf:
            nearend_files = [nearend_map[s] for s in common_nf]
            farend_files  = [farend_map[s]  for s in common_nf]
            echo_files    = []  # no echo files matched
        else:
            # Last resort: pair by sorted index
            n = min(len(nearend_all), len(farend_all))
            nearend_files = nearend_all[:n]
            farend_files  = farend_all[:n]
            echo_files    = echo_all[:n] if len(echo_all) >= n else []
    else:
        nearend_files = [nearend_map[s] for s in common]
        farend_files  = [farend_map[s]  for s in common]
        echo_files    = [echo_map[s]    for s in common]

    return nearend_files, farend_files, echo_files


def make_aec_dataset(aec_root, noise_dir=None, rir_dir=None,
                     use_pregenerated_echo=True, **kwargs):
    """
    Build a SpeechEnhancementDataset from AEC-Challenge synthetic data.

    Directory layout:
        aec_root/
          ├── farend_speech/        # far-end signals
          ├── echo_signal/          # pre-generated echo signals
          ├── nearend_speech/       # clean near-end targets
          ├── nearend_mic_signal/   # pre-mixed mic (not used for on-the-fly mixing)
          └── meta.csv              # metadata

    Files linked by name: fileid_{int}.wav

    Args:
        aec_root: path to AEC synthetic directory
        noise_dir: shared noise directory (optional)
        rir_dir: shared RIR directory (optional, synthetic if omitted)
        use_pregenerated_echo: if True, use echo_signal/ files directly;
                               if False, generate echo on-the-fly from farend
    """
    nearend_files, farend_files, echo_files = _build_aec_file_lists(aec_root)
    noise_files = _scan_wavs(noise_dir) if noise_dir else []
    rir_files   = _scan_wavs(rir_dir) if rir_dir else []

    meta = _parse_aec_meta(aec_root)
    if meta:
        print(f"[make_aec_dataset] Loaded meta.csv with {len(meta)} entries")

    print(f"[make_aec_dataset] {len(nearend_files)} nearend, "
          f"{len(farend_files)} farend, {len(echo_files)} echo")

    return SpeechEnhancementDataset(
        nearend_files=nearend_files,
        farend_files=farend_files,
        echo_files=echo_files,
        noise_files=noise_files,
        rir_files=rir_files,
        use_pregenerated_echo=use_pregenerated_echo,
        **kwargs,
    )


def make_dns_dataset(dns_root, noise_dir=None, rir_dir=None, **kwargs):
    """
    Build a SpeechEnhancementDataset from DNS-Challenge data.

    Supports two layouts:
      Standard:
        dns_root/
          └── clean/                # clean near-end speech

      Multi-directory (e.g. DNS-Challenge datasets/dns/):
        dns_root/
          ├── datasets.clean.emotional_speech/
          ├── datasets.clean.french_data/
          └── ...                   # all dirs starting with 'datasets.clean'

    Priority:
      1. dns_root/clean/ if it exists
      2. All subdirs starting with 'datasets.clean'
      3. dns_root itself (fallback)

    No far-end → echo step is skipped.
    """
    clean_subdir = os.path.join(dns_root, "clean")
    if os.path.isdir(clean_subdir):
        nearend_files = _scan_wavs(clean_subdir)
    else:
        clean_dirs = [
            os.path.join(dns_root, d)
            for d in sorted(os.listdir(dns_root))
            if d.startswith("datasets.clean") and os.path.isdir(os.path.join(dns_root, d))
        ]
        if clean_dirs:
            print(f"[make_dns_dataset] Found {len(clean_dirs)} datasets.clean.* dirs: "
                  f"{[os.path.basename(d) for d in clean_dirs]}")
            nearend_files = _scan_wavs(clean_dirs)
        else:
            nearend_files = _scan_wavs(dns_root)

    noise_files   = _scan_wavs(noise_dir) if noise_dir else []
    rir_files     = _scan_wavs(rir_dir) if rir_dir else []

    return SpeechEnhancementDataset(
        nearend_files=nearend_files,
        farend_files=None,
        noise_files=noise_files,
        rir_files=rir_files,
        **kwargs,
    )


# ─── Combined dataset ─────────────────────────────────────────────────────────

class CombinedDataset(Dataset):
    """Concatenation of multiple datasets."""

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.cum_lengths = []
        s = 0
        for d in datasets:
            s += len(d)
            self.cum_lengths.append(s)
        total = self.cum_lengths[-1] if self.cum_lengths else 0
        print(f"[CombinedDataset] {len(datasets)} sub-datasets, total={total} samples")

    def __len__(self):
        return self.cum_lengths[-1] if self.cum_lengths else 0

    def __getitem__(self, idx):
        for i, cum in enumerate(self.cum_lengths):
            if idx < cum:
                offset = self.cum_lengths[i - 1] if i > 0 else 0
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range")
