"""
Inference script for DeepVQE.

Supports three modes:
  1. Offline  – process entire files at once (best quality)
  2. Stream   – chunk-based overlap-add (low latency, bounded memory)
  3. Live     – real-time mic → model → speaker (requires sounddevice)

Usage examples
──────────────
  # Offline – single pair
  python inference.py --mic mic.wav --ref ref.wav --out enhanced.wav --ckpt best.pt

  # Offline – directory of files (mic_dir/ and ref_dir/ with matching names)
  python inference.py --mic_dir mic_dir --ref_dir ref_dir --out_dir enh_dir --ckpt best.pt

  # Stream – chunk-based
  python inference.py --mic mic.wav --ref ref.wav --out enhanced.wav --ckpt best.pt --stream

  # Live – real-time mic → speaker (denoise only)
  python inference.py --live --ckpt best.pt

  # Live – with far-end reference file for AEC
  python inference.py --live --ref ref.wav --ckpt best.pt

  # Live – list available audio devices
  python inference.py --live --list_devices

  # Mic-only (no ref / denoising only)
  python inference.py --mic mic.wav --out enhanced.wav --ckpt best.pt
"""

import os
import sys
import glob
import time
import argparse

import numpy as np
import torch

from utils import load_wav, save_wav, stft, istft, SAMPLE_RATE, N_FFT, HOP_LENGTH
from cmy_deepvqe import DeepVQE


# ─── Offline inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def infer_offline(model, mic_wav, ref_wav, n_fft=N_FFT, hop=HOP_LENGTH, device="cpu"):
    """
    Run the full model on complete waveforms.

    Args:
        model: DeepVQE model (eval mode)
        mic_wav: np.ndarray (L,)  microphone signal
        ref_wav: np.ndarray (L,)  far-end reference (same length as mic_wav)
        n_fft, hop: STFT parameters
        device: torch device string

    Returns:
        enhanced: np.ndarray (L,)
    """
    orig_len = len(mic_wav)

    mic_t = torch.from_numpy(mic_wav).float().unsqueeze(0).to(device)   # (1, L)
    ref_t = torch.from_numpy(ref_wav).float().unsqueeze(0).to(device)

    mic_spec = stft(mic_t, n_fft, hop)            # (1, F, T, 2)
    ref_spec = stft(ref_t, n_fft, hop)

    enh_spec = model(mic_spec, ref_spec)           # (1, F, T, 2)

    enh_wav = istft(enh_spec, n_fft, hop)          # (1, L')
    enh_wav = enh_wav.squeeze(0).cpu().numpy()

    # Trim / pad to original length
    if len(enh_wav) >= orig_len:
        enh_wav = enh_wav[:orig_len]
    else:
        enh_wav = np.pad(enh_wav, (0, orig_len - len(enh_wav)))

    return enh_wav


# ─── Stream (chunk-based overlap-add) inference ───────────────────────────────

@torch.no_grad()
def infer_stream(model, mic_wav, ref_wav,
                 n_fft=N_FFT, hop=HOP_LENGTH, chunk_sec=1.0, sr=SAMPLE_RATE,
                 device="cpu"):
    """
    Chunk-based overlap-add streaming inference.

    The audio is split into overlapping chunks.  Each chunk is processed
    independently through STFT → model → iSTFT, then stitched back
    together with overlap-add so that the result matches the offline
    output up to boundary artefacts.

    Parameters
    ----------
    model     : DeepVQE (eval mode)
    mic_wav   : np.ndarray (L,)
    ref_wav   : np.ndarray (L,)
    chunk_sec : duration of each processing chunk in seconds
    sr        : sample rate
    device    : torch device string

    Returns
    -------
    enhanced  : np.ndarray (L,)
    """
    orig_len = len(mic_wav)
    chunk_samples = int(chunk_sec * sr)

    # Overlap = n_fft samples (one STFT window) to avoid boundary artefacts
    overlap = n_fft

    # Ensure chunk is at least 2× overlap for meaningful processing
    if chunk_samples < 2 * overlap:
        chunk_samples = 2 * overlap

    step = chunk_samples - overlap          # advance per chunk

    # Pad signals to make them divisible by step size
    pad_len = 0
    if (orig_len - overlap) % step != 0:
        pad_len = step - ((orig_len - overlap) % step)
    mic_padded = np.pad(mic_wav, (0, pad_len))
    ref_padded = np.pad(ref_wav, (0, pad_len))
    total_len = len(mic_padded)

    # Output buffer + normalization window (for overlap-add weighting)
    out_buf = np.zeros(total_len, dtype=np.float32)
    win_buf = np.zeros(total_len, dtype=np.float32)

    # Synthesis window for OLA (Hann)
    syn_win = np.hanning(chunk_samples).astype(np.float32)

    pos = 0
    chunk_idx = 0
    while pos + chunk_samples <= total_len:
        mic_chunk = mic_padded[pos: pos + chunk_samples]
        ref_chunk = ref_padded[pos: pos + chunk_samples]

        mic_t = torch.from_numpy(mic_chunk).float().unsqueeze(0).to(device)
        ref_t = torch.from_numpy(ref_chunk).float().unsqueeze(0).to(device)

        mic_spec = stft(mic_t, n_fft, hop)
        ref_spec = stft(ref_t, n_fft, hop)

        enh_spec = model(mic_spec, ref_spec)
        enh_chunk = istft(enh_spec, n_fft, hop).squeeze(0).cpu().numpy()

        # Trim/pad to chunk_samples
        if len(enh_chunk) < chunk_samples:
            enh_chunk = np.pad(enh_chunk, (0, chunk_samples - len(enh_chunk)))
        else:
            enh_chunk = enh_chunk[:chunk_samples]

        # Apply synthesis window and overlap-add
        enh_chunk *= syn_win
        out_buf[pos: pos + chunk_samples] += enh_chunk
        win_buf[pos: pos + chunk_samples] += syn_win

        pos += step
        chunk_idx += 1

    # Normalise by the overlap-add window sum
    nonzero = win_buf > 1e-8
    out_buf[nonzero] /= win_buf[nonzero]

    return out_buf[:orig_len]


# ─── Live mic → model → speaker ───────────────────────────────────────────────

@torch.no_grad()
def infer_live(model, n_fft=N_FFT, hop=HOP_LENGTH, chunk_sec=0.5,
               sr=SAMPLE_RATE, device="cpu", ref_wav=None,
               input_device=None, output_device=None, save_path=None):
    """
    Real-time inference: capture from microphone, enhance, play to speaker.

    Uses chunk-based overlap-add processing.  Each chunk is captured from the
    mic, paired with a reference signal (zeros if no ref), processed by the
    model, and immediately played back through the speaker.

    Parameters
    ----------
    model          : DeepVQE in eval mode
    n_fft, hop     : STFT parameters
    chunk_sec      : processing chunk length in seconds
    sr             : sample rate
    device         : torch device
    ref_wav        : optional np.ndarray – far-end reference (looped)
    input_device   : sounddevice input device index (None = default)
    output_device  : sounddevice output device index (None = default)
    save_path      : if set, also save the enhanced output to this wav file
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] Live mode requires the 'sounddevice' package.")
        print("        Install it with:  pip install sounddevice")
        sys.exit(1)

    chunk_samples = int(chunk_sec * sr)
    overlap = n_fft            # overlap to avoid STFT boundary artefacts
    if chunk_samples < 2 * overlap:
        chunk_samples = 2 * overlap
    step = chunk_samples - overlap

    # Persistent buffers for the overlap region
    mic_buffer = np.zeros(overlap, dtype=np.float32)
    ref_buffer = np.zeros(overlap, dtype=np.float32)
    ref_pos = 0                # current read position inside ref_wav

    # Collect all enhanced audio for optional saving
    recorded_chunks = [] if save_path else None

    print(f"[live] sr={sr}, chunk={chunk_samples} samples ({chunk_sec:.2f}s), "
          f"overlap={overlap}, step={step}")
    print(f"[live] input_device={input_device}, output_device={output_device}")
    if ref_wav is not None:
        print(f"[live] Using reference signal ({len(ref_wav)/sr:.1f}s, looped)")
    else:
        print("[live] No reference – denoise-only mode")
    print("[live] Press Ctrl+C to stop\n")

    try:
        with sd.Stream(samplerate=sr, blocksize=step,
                       channels=1, dtype="float32",
                       device=(input_device, output_device)) as stream:
            while True:
                # ── 1. Read `step` new samples from microphone ────────────
                audio_in, overflowed = stream.read(step)
                if overflowed:
                    print("[live] WARNING: input overflow")
                mic_new = audio_in[:, 0]  # (step,)

                # Build full chunk = [overlap tail | new samples]
                mic_chunk = np.concatenate([mic_buffer, mic_new])
                mic_buffer = mic_chunk[-overlap:]     # save tail for next iter

                # ── 2. Build reference chunk ───────────────────────────────
                if ref_wav is not None:
                    # Read `step` samples from ref, looping if needed
                    ref_new = np.zeros(step, dtype=np.float32)
                    filled = 0
                    rp = ref_pos
                    while filled < step:
                        avail = min(len(ref_wav) - rp, step - filled)
                        ref_new[filled:filled + avail] = ref_wav[rp:rp + avail]
                        filled += avail
                        rp = (rp + avail) % len(ref_wav)
                    ref_pos = rp
                    ref_chunk = np.concatenate([ref_buffer, ref_new])
                    ref_buffer = ref_chunk[-overlap:]
                else:
                    ref_chunk = np.zeros(chunk_samples, dtype=np.float32)

                # ── 3. STFT → model → iSTFT ───────────────────────────────
                mic_t = torch.from_numpy(mic_chunk).float().unsqueeze(0).to(device)
                ref_t = torch.from_numpy(ref_chunk).float().unsqueeze(0).to(device)

                mic_spec = stft(mic_t, n_fft, hop)
                ref_spec = stft(ref_t, n_fft, hop)
                enh_spec = model(mic_spec, ref_spec)
                enh_wav = istft(enh_spec, n_fft, hop).squeeze(0).cpu().numpy()

                # ── 4. Extract the `step` new samples (discard overlap) ───
                if len(enh_wav) >= chunk_samples:
                    out_samples = enh_wav[overlap:overlap + step]
                elif len(enh_wav) >= step:
                    out_samples = enh_wav[-step:]
                else:
                    out_samples = np.pad(enh_wav, (0, step - len(enh_wav)))

                # ── 5. Write to speaker ───────────────────────────────────
                stream.write(out_samples.reshape(-1, 1))

                if recorded_chunks is not None:
                    recorded_chunks.append(out_samples.copy())

    except KeyboardInterrupt:
        print("\n[live] Stopped by user")

    # Optionally save the recorded output
    if recorded_chunks and save_path:
        full = np.concatenate(recorded_chunks)
        save_wav(save_path, full, sr)
        print(f"[live] Saved {len(full)/sr:.1f}s → {save_path}")


# ─── File helpers ──────────────────────────────────────────────────────────────

def _match_length(mic, ref):
    """Pad/trim ref to match mic length."""
    if len(ref) >= len(mic):
        return ref[:len(mic)]
    return np.pad(ref, (0, len(mic) - len(ref)))


def _scan_wavs(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True))
    return files


def _process_pair(model, mic_path, ref_path, out_path, args, device):
    """Process a single mic/ref pair and save the result."""
    sr = args.sr
    mic_wav = load_wav(mic_path, sr)

    if ref_path is not None and os.path.isfile(ref_path):
        ref_wav = load_wav(ref_path, sr)
        ref_wav = _match_length(mic_wav, ref_wav)
    else:
        ref_wav = np.zeros_like(mic_wav)

    t0 = time.perf_counter()

    if args.stream:
        enh = infer_stream(
            model, mic_wav, ref_wav,
            n_fft=args.n_fft, hop=args.hop_length,
            chunk_sec=args.chunk_sec, sr=sr, device=device,
        )
    else:
        enh = infer_offline(
            model, mic_wav, ref_wav,
            n_fft=args.n_fft, hop=args.hop_length, device=device,
        )

    elapsed = time.perf_counter() - t0
    duration = len(mic_wav) / sr
    rtf = elapsed / duration

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_wav(out_path, enh, sr)

    return duration, elapsed, rtf


# ─── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, device="cpu"):
    """
    Load a DeepVQE model from a checkpoint file.

    Supports two checkpoint formats:
      1. Full training checkpoint  (dict with 'model_state_dict' key)
      2. Plain state_dict
    """
    model = DeepVQE()

    if ckpt_path is not None and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[load_model] Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
        elif isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            print(f"[load_model] Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
        else:
            model.load_state_dict(ckpt)
            print(f"[load_model] Loaded state_dict from {ckpt_path}")
    else:
        print("[load_model] WARNING: No checkpoint provided – using random weights")

    model = model.to(device).eval()
    return model


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DeepVQE inference – offline & stream modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── input / output ─────────────────────────────
    g_io = p.add_argument_group("I/O – single file")
    g_io.add_argument("--mic", type=str, default=None,
                      help="Path to microphone .wav file")
    g_io.add_argument("--ref", type=str, default=None,
                      help="Path to far-end reference .wav file (omit for denoise-only)")
    g_io.add_argument("--out", type=str, default=None,
                      help="Output enhanced .wav path")

    g_dir = p.add_argument_group("I/O – directory batch")
    g_dir.add_argument("--mic_dir", type=str, default=None,
                       help="Directory of microphone .wav files")
    g_dir.add_argument("--ref_dir", type=str, default=None,
                       help="Directory of far-end .wav files (matched by filename)")
    g_dir.add_argument("--out_dir", type=str, default=None,
                       help="Output directory for enhanced files")

    # ── model ──────────────────────────────────────
    g_model = p.add_argument_group("Model")
    g_model.add_argument("--ckpt", type=str, default=None,
                         help="Path to model checkpoint (.pt)")
    g_model.add_argument("--device", type=str, default=None,
                         help="Device (default: cuda if available, else cpu)")

    # ── audio ──────────────────────────────────────
    g_audio = p.add_argument_group("Audio")
    g_audio.add_argument("--sr", type=int, default=SAMPLE_RATE)
    g_audio.add_argument("--n_fft", type=int, default=N_FFT)
    g_audio.add_argument("--hop_length", type=int, default=HOP_LENGTH)

    # ── mode ───────────────────────────────────────
    g_mode = p.add_argument_group("Inference mode")
    g_mode.add_argument("--stream", action="store_true", default=False,
                        help="Enable chunk-based streaming inference")
    g_mode.add_argument("--chunk_sec", type=float, default=1.0,
                        help="Chunk duration in seconds for stream mode (default: 1.0)")

    # ── live mode ──────────────────────────────────
    g_live = p.add_argument_group("Live mode (mic → model → speaker)")
    g_live.add_argument("--live", action="store_true", default=False,
                        help="Real-time mic → model → speaker")
    g_live.add_argument("--list_devices", action="store_true", default=False,
                        help="List available audio devices and exit")
    g_live.add_argument("--input_device", type=int, default=None,
                        help="Input (mic) device index (see --list_devices)")
    g_live.add_argument("--output_device", type=int, default=None,
                        help="Output (speaker) device index (see --list_devices)")
    g_live.add_argument("--save_live", type=str, default=None,
                        help="Save the live-enhanced output to this .wav file")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ─────────────────────────────────────
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[inference] device={device}, stream={args.stream}, live={args.live}")

    # ── List audio devices ─────────────────────────
    if args.list_devices:
        try:
            import sounddevice as sd
            print(sd.query_devices())
        except ImportError:
            print("[ERROR] 'sounddevice' is required.  pip install sounddevice")
        return

    # ── Load model ─────────────────────────────────
    model = load_model(args.ckpt, device)

    # ── Live mode ──────────────────────────────────
    if args.live:
        ref_wav = None
        if args.ref is not None and os.path.isfile(args.ref):
            ref_wav = load_wav(args.ref, args.sr)
            print(f"[live] Loaded reference: {args.ref} ({len(ref_wav)/args.sr:.1f}s)")

        infer_live(
            model,
            n_fft=args.n_fft, hop=args.hop_length,
            chunk_sec=args.chunk_sec, sr=args.sr,
            device=device, ref_wav=ref_wav,
            input_device=args.input_device,
            output_device=args.output_device,
            save_path=args.save_live,
        )
        return

    mode_tag = "stream" if args.stream else "offline"

    # ── Single file mode ───────────────────────────
    if args.mic is not None:
        if args.out is None:
            base, ext = os.path.splitext(args.mic)
            args.out = f"{base}_enhanced{ext}"

        dur, elapsed, rtf = _process_pair(model, args.mic, args.ref, args.out, args, device)
        print(f"[{mode_tag}] {args.mic}")
        print(f"  duration={dur:.2f}s  time={elapsed:.3f}s  RTF={rtf:.4f}")
        print(f"  saved → {args.out}")
        return

    # ── Directory batch mode ───────────────────────
    if args.mic_dir is not None:
        if args.out_dir is None:
            args.out_dir = args.mic_dir + "_enhanced"
        os.makedirs(args.out_dir, exist_ok=True)

        mic_files = _scan_wavs(args.mic_dir)
        if not mic_files:
            print(f"[ERROR] No .wav files found in {args.mic_dir}")
            sys.exit(1)

        total_dur = 0.0
        total_time = 0.0

        for i, mic_path in enumerate(mic_files):
            # Match ref by filename stem
            rel = os.path.relpath(mic_path, args.mic_dir)
            out_path = os.path.join(args.out_dir, rel)

            ref_path = None
            if args.ref_dir is not None:
                ref_path = os.path.join(args.ref_dir, rel)

            dur, elapsed, rtf = _process_pair(model, mic_path, ref_path, out_path, args, device)
            total_dur += dur
            total_time += elapsed

            print(f"  [{i+1}/{len(mic_files)}] {rel}  "
                  f"dur={dur:.2f}s  time={elapsed:.3f}s  RTF={rtf:.4f}")

        avg_rtf = total_time / total_dur if total_dur > 0 else 0
        print(f"\n[{mode_tag}] Done – {len(mic_files)} files, "
              f"total_dur={total_dur:.1f}s, total_time={total_time:.1f}s, "
              f"avg_RTF={avg_rtf:.4f}")
        print(f"  saved → {args.out_dir}/")
        return

    # ── No input specified ─────────────────────────
    print("[ERROR] Specify --mic (single file), --mic_dir (batch), or --live. "
          "Run with --help for usage.")
    sys.exit(1)


if __name__ == "__main__":
    main()
