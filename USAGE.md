# DeepVQE — Training & Inference Guide

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
  - [Offline (single file)](#offline-single-file)
  - [Offline (batch directory)](#offline-batch-directory)
  - [Stream (chunk-based)](#stream-chunk-based)
  - [Live (real-time mic → speaker)](#live-real-time-mic--speaker)
- [Configuration Reference](#configuration-reference)
- [Loss Functions](#loss-functions)
- [Tips & FAQ](#tips--faq)

---

## Project Structure

```
deepvqe/
├── cmy_deepvqe.py      # DeepVQE model definition
├── dataset.py           # Unified dataset (on-the-fly mixing, 8-step pipeline)
├── loss.py              # CompressedSpecLoss + SISNRLoss + CombinedLoss
├── config.py            # All hyperparameters in a single dataclass
├── utils.py             # STFT / iSTFT, audio I/O, SI-SNR / SNR metrics
├── train.py             # Full training script with CLI
├── inference.py         # Offline / stream / live inference
└── USAGE.md             # ← you are here
```

---

## Installation

```bash
pip install torch torchaudio einops scipy soundfile numpy
# Optional – for live inference:
pip install sounddevice
# Optional – for FLOPS counting in cmy_deepvqe.py __main__:
pip install ptflops
```

> **Python ≥ 3.8** and **PyTorch ≥ 1.11** are recommended.

---

## Data Preparation

The training pipeline supports two public datasets. You can use either or both.

### AEC-Challenge (Acoustic Echo Cancellation)

Download the [AEC-Challenge synthetic dataset](https://github.com/microsoft/AEC-Challenge).
The expected directory layout:

```
aec_root/
├── farend_speech/          # far-end signals (fileid_0.wav, fileid_1.wav, …)
├── echo_signal/            # pre-generated echo signals
├── nearend_speech/         # clean near-end target
├── nearend_mic_signal/     # mic = nearend + echo (± noise)  [not used for on-the-fly mixing]
└── meta.csv                # fileid, ser, nearend_scale, is_nearend_noisy, is_farend_noisy
```

Files are linked by name: all four directories contain `fileid_{int}.wav` with the same integer.

### DNS-Challenge (Deep Noise Suppression)

Download clean speech from the [DNS-Challenge](https://github.com/microsoft/DNS-Challenge).

```
dns_root/
└── clean/                  # clean near-end speech (*.wav)
```

Since DNS has no far-end signal, the echo step is skipped automatically.

### Shared directories (optional)

```
noise_dir/                  # background noise clips (*.wav)
rir_dir/                    # room impulse responses (*.wav)
                            # if omitted, synthetic RIRs are generated on-the-fly
```

---

## Training

### Quick start

```bash
# AEC data only
python train.py \
    --aec_root /data/aec_challenge/synthetic \
    --noise_dir /data/noise \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3

# DNS data only
python train.py \
    --dns_root /data/dns_challenge \
    --noise_dir /data/noise \
    --epochs 100

# Both AEC + DNS combined
python train.py \
    --aec_root /data/aec_challenge/synthetic \
    --dns_root /data/dns_challenge \
    --noise_dir /data/noise \
    --rir_dir /data/rir \
    --val_aec_root /data/aec_challenge/synthetic_val \
    --val_dns_root /data/dns_challenge_val \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --save_dir checkpoints
```

### Resume from checkpoint

```bash
python train.py \
    --aec_root /data/aec_challenge/synthetic \
    --noise_dir /data/noise \
    --resume checkpoints/epoch_50.pt
```

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--aec_root` | None | AEC-Challenge synthetic root directory |
| `--dns_root` | None | DNS-Challenge root directory |
| `--noise_dir` | None | Shared noise directory |
| `--rir_dir` | None | Shared RIR directory (synthetic if omitted) |
| `--val_aec_root` | None | Validation AEC root |
| `--val_dns_root` | None | Validation DNS root |
| `--use_pregenerated_echo` | True | Use `echo_signal/` directly |
| `--no_pregenerated_echo` | — | Generate echo on-the-fly instead |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1e-5 | AdamW weight decay |
| `--grad_clip` | 5.0 | Gradient clipping norm |
| `--lr_scheduler` | cosine | `cosine` or `step` |
| `--segment_len` | 4.0 | Training clip length (seconds) |
| `--sr` | 16000 | Sample rate |
| `--n_fft` | 512 | FFT size |
| `--hop_length` | 256 | STFT hop length |
| `--lambda_spec` | 1.0 | Weight for spectral loss |
| `--lambda_sisnr` | 0.1 | Weight for SI-SNR loss |
| `--save_dir` | checkpoints | Where to save checkpoints |
| `--resume` | None | Path to checkpoint to resume |
| `--num_workers` | 4 | DataLoader workers |
| `--seed` | 42 | Random seed |
| `--device` | cuda | `cuda` or `cpu` |
| `--log_interval` | 50 | Print every N steps |

### Checkpoints

Checkpoints are saved to `--save_dir` as `epoch_{N}.pt` and `best.pt`.
Each checkpoint contains:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "best_val_loss": float,
}
```

---

## Inference

The inference script supports three modes: **offline**, **stream**, and **live**.

### Offline (single file)

Process a complete wav file pair at once. Best quality, higher latency.

```bash
# AEC: mic + far-end reference
python inference.py \
    --mic mic.wav \
    --ref ref.wav \
    --out enhanced.wav \
    --ckpt checkpoints/best.pt

# Denoise only (no reference)
python inference.py \
    --mic noisy.wav \
    --out enhanced.wav \
    --ckpt checkpoints/best.pt
```

If `--out` is omitted, the output is saved as `<mic_name>_enhanced.wav`.

### Offline (batch directory)

Process a directory of wav files. Files in `--mic_dir` and `--ref_dir` are matched by filename.

```bash
python inference.py \
    --mic_dir test_mic/ \
    --ref_dir test_ref/ \
    --out_dir test_enhanced/ \
    --ckpt checkpoints/best.pt
```

### Stream (chunk-based)

Chunk-based overlap-add processing. Lower latency than offline, suitable for near-real-time use.

```bash
python inference.py \
    --mic mic.wav \
    --ref ref.wav \
    --out enhanced.wav \
    --ckpt checkpoints/best.pt \
    --stream \
    --chunk_sec 1.0
```

`--chunk_sec` controls the chunk duration. Smaller = lower latency but potentially more boundary artefacts.

### Live (real-time mic → speaker)

Captures audio from your microphone, enhances it in real-time, and plays through the speaker.

> **Requires:** `pip install sounddevice`

```bash
# List available audio devices
python inference.py --live --list_devices

# Denoise only (no reference)
python inference.py --live --ckpt checkpoints/best.pt

# AEC with far-end reference (looped)
python inference.py --live --ref ref.wav --ckpt checkpoints/best.pt

# Select specific mic/speaker + save output
python inference.py --live \
    --ckpt checkpoints/best.pt \
    --input_device 1 \
    --output_device 3 \
    --chunk_sec 0.5 \
    --save_live live_output.wav
```

Press `Ctrl+C` to stop live inference.

### Inference flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mic` | None | Input microphone wav file |
| `--ref` | None | Far-end reference wav (omit for denoise-only) |
| `--out` | None | Output wav path (auto-generated if omitted) |
| `--mic_dir` | None | Directory of mic wav files (batch mode) |
| `--ref_dir` | None | Directory of ref wav files (batch mode) |
| `--out_dir` | None | Output directory (batch mode) |
| `--ckpt` | None | Model checkpoint path |
| `--device` | auto | `cuda` if available, else `cpu` |
| `--sr` | 16000 | Sample rate |
| `--n_fft` | 512 | FFT size |
| `--hop_length` | 256 | STFT hop length |
| `--stream` | False | Enable chunk-based streaming |
| `--chunk_sec` | 1.0 | Chunk duration in seconds |
| `--live` | False | Enable real-time mic → speaker |
| `--list_devices` | False | List audio devices and exit |
| `--input_device` | None | Mic device index |
| `--output_device` | None | Speaker device index |
| `--save_live` | None | Save live output to wav file |

---

## Configuration Reference

All hyperparameters are defined in `config.py` as a `@dataclass`. The training script maps CLI arguments to this config. You can also import and modify `Config` directly in custom scripts:

```python
from config import Config

cfg = Config(
    aec_root="/data/aec",
    noise_dir="/data/noise",
    epochs=200,
    batch_size=16,
    lr=5e-4,
)
```

### Data pipeline (dataset.py)

Each training sample is generated on-the-fly through an 8-step pipeline:

1. **Load** random near-end speech + far-end signal
2. **Sample** random parameters (SNR, SER, gain, RT60)
3. **Reverb** — apply RIR to near-end speech
4. **Echo** — convolve far-end with echo path (or use pre-generated echo)
5. **Noise** — add background noise at random SNR
6. **Compose** — `mic = reverb_nearend + echo + noise`
7. **Gain** — apply random gain
8. **Normalize** — peak-normalize to prevent clipping

Key mixing parameters (all configurable via CLI or `Config`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `snr_range` | (-5, 20) dB | Near-end to noise SNR |
| `ser_range` | (-10, 10) dB | Signal to echo ratio |
| `gain_range_db` | (-6, 6) dB | Random gain |
| `rt60_range` | (0.1, 0.7) s | Synthetic RIR RT60 |
| `echo_delay_range_ms` | (1, 200) ms | Echo path delay |
| `echo_rt60_range` | (0.05, 0.3) s | Echo path RT60 |
| `reverb_prob` | 0.5 | Probability of applying reverb |
| `echo_prob` | 1.0 | Probability of adding echo |
| `noise_prob` | 1.0 | Probability of adding noise |

---

## Loss Functions

The training uses a combined loss (`loss.py`):

```
total_loss = λ_spec × CompressedSpecLoss + λ_sisnr × SISNRLoss
```

- **CompressedSpecLoss** — operates in the STFT domain with power compression (c=0.3). Balances complex spectral distance (α=0.7) and magnitude distance (1−α=0.3).
- **SISNRLoss** — negative scale-invariant SNR in the time domain (via iSTFT).
- **CombinedLoss** — weighted sum of the above (default: λ_spec=1.0, λ_sisnr=0.1).

---

## Tips & FAQ

### How much VRAM do I need?

With `batch_size=8`, `segment_len=4.0s`, `n_fft=512`: approximately **6–8 GB**. Reduce `batch_size` or `segment_len` if you run out of memory.

### Can I train on CPU?

Yes, pass `--device cpu`. It will be slow but functional.

### How do I use only the AEC dataset?

```bash
python train.py --aec_root /path/to/aec --noise_dir /path/to/noise
```

### How do I use only DNS (denoising, no echo)?

```bash
python train.py --dns_root /path/to/dns --noise_dir /path/to/noise
```

The echo step is automatically skipped when no far-end files are present.

### What if I don't have RIR files?

Omit `--rir_dir`. The dataset will generate synthetic RIRs on-the-fly using exponentially decaying noise.

### Pre-generated echo vs on-the-fly echo

By default (`--use_pregenerated_echo`), the AEC dataset uses the `echo_signal/` files directly from the AEC-Challenge data. To generate echo on-the-fly by convolving the far-end with a random echo path, use:

```bash
python train.py --aec_root /data/aec --no_pregenerated_echo
```

### Live mode latency

The `--chunk_sec` flag controls the trade-off:
- **0.25s** — ~250ms latency, more CPU-intensive
- **0.5s** — ~500ms latency, balanced (recommended)
- **1.0s** — ~1s latency, most stable

### Model architecture

DeepVQE is a U-Net style encoder-decoder with:
- Dual-path encoder (mic + ref branches)
- Attention-based alignment block (AlignBlock)
- GRU bottleneck
- Sub-pixel convolution decoder
- Complex convolving mask (CCM) output

Default channel config: mic=[64,128,128,128,128], ref=[32,128], dec=[128,128,128,64,27], GRU hidden=256.
