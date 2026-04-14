"""
Loss functions for DeepVQE training.

Combines:
  - Complex compressed MSE on spectrogram (magnitude + phase aware)
  - Time-domain SI-SNR loss
"""

import os

import torch
import torch.nn as nn
import soundfile as sf

from utils import istft


class MagLoss(nn.Module):
    """
    Compressed magnitude MSE on a single STFT scale.
    L = mean( ||Y|^c - |S|^c|^2 )
    Operates on (B, F, T, 2) real-valued STFT tensors.
    """

    def __init__(self, compress=0.3):
        super().__init__()
        self.c = compress

    def forward(self, est_spec, tgt_spec):
        est_mag = torch.sqrt(est_spec[..., 0] ** 2 + est_spec[..., 1] ** 2 + 1e-8)
        tgt_mag = torch.sqrt(tgt_spec[..., 0] ** 2 + tgt_spec[..., 1] ** 2 + 1e-8)
        return torch.mean((est_mag.pow(self.c) - tgt_mag.pow(self.c)) ** 2)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss: spectral convergence averaged over multiple scales.
    L = mean over scales of  || |Y| - |S| ||_F / || |S| ||_F
    Operates on (B, L) time-domain waveforms.
    Scales: n_fft = [256, 512, 1024], hop = [64, 128, 256]
    """

    def __init__(
        self,
        fft_sizes=(256, 512, 1024),
        hop_sizes=(64, 128, 256),
        win_sizes=(256, 512, 1024),
        eps=1e-8,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.scales = list(zip(fft_sizes, hop_sizes, win_sizes))
        self.eps = eps

    def _stft_mag(self, wav, n_fft, hop, win):
        window = torch.hann_window(win, device=wav.device)
        X = torch.stft(wav, n_fft, hop, win, window=window, return_complex=False)
        return torch.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2 + self.eps)  # (B, F, T)

    def forward(self, est_wav, tgt_wav):
        loss = est_wav.new_zeros(1)
        for n_fft, hop, win in self.scales:
            est_mag = self._stft_mag(est_wav, n_fft, hop, win)
            tgt_mag = self._stft_mag(tgt_wav, n_fft, hop, win)
            diff_norm = torch.norm(est_mag - tgt_mag, p="fro", dim=(-2, -1))
            tgt_norm  = torch.norm(tgt_mag,           p="fro", dim=(-2, -1)).clamp(min=self.eps)
            loss = loss + (diff_norm / tgt_norm).mean()
        return loss / len(self.scales)


class SISNRLoss(nn.Module):
    """Negative SI-SNR loss in time domain."""

    def forward(self, est_wav, tgt_wav):
        """
        est_wav, tgt_wav: (B, L) time-domain waveforms.
        Returns negative SI-SNR (to minimize).
        """
        # Zero-mean
        est_wav = est_wav - est_wav.mean(dim=-1, keepdim=True)
        tgt_wav = tgt_wav - tgt_wav.mean(dim=-1, keepdim=True)

        dot = torch.sum(est_wav * tgt_wav, dim=-1, keepdim=True)
        s_ref_energy = torch.sum(tgt_wav ** 2, dim=-1, keepdim=True) + 1e-8
        proj = dot * tgt_wav / s_ref_energy

        noise = est_wav - proj
        proj_power  = torch.sum(proj  ** 2, dim=-1).clamp(min=1e-8)
        noise_power = torch.sum(noise ** 2, dim=-1).clamp(min=1e-8)
        si_snr = 10 * torch.log10((proj_power / noise_power).clamp(min=1e-10, max=1e10))
        return -si_snr.mean()


class CombinedLoss(nn.Module):
    """
    total = lambda_mag     * MagLoss                (compressed magnitude MSE, single scale)
          + lambda_mrsstft * MultiResolutionSTFTLoss (spectral convergence, multi-scale)
          + lambda_sisnr   * SISNRLoss               (time-domain SI-SNR)
    """

    def __init__(self, compress=0.3,
                 lambda_mag=1.0, lambda_mrsstft=1.0, lambda_sisnr=0.1,
                 n_fft=512, hop_length=256,
                 mrsstft_fft_sizes=(256, 512, 1024),
                 mrsstft_hop_sizes=(64, 128, 256),
                 mrsstft_win_sizes=(256, 512, 1024)):
        super().__init__()
        self.mag_loss     = MagLoss(compress=compress)
        self.mrsstft_loss = MultiResolutionSTFTLoss(
            fft_sizes=mrsstft_fft_sizes,
            hop_sizes=mrsstft_hop_sizes,
            win_sizes=mrsstft_win_sizes,
        )
        self.sisnr_loss   = SISNRLoss()
        self.lambda_mag     = lambda_mag
        self.lambda_mrsstft = lambda_mrsstft
        self.lambda_sisnr   = lambda_sisnr
        self._n_fft = n_fft
        self._hop   = hop_length

    def forward(self, est_spec, tgt_spec):
        est_wav = istft(est_spec, self._n_fft, self._hop)
        tgt_wav = istft(tgt_spec, self._n_fft, self._hop)
        min_len = min(est_wav.shape[-1], tgt_wav.shape[-1])
        est_wav = est_wav[..., :min_len]
        tgt_wav = tgt_wav[..., :min_len]

        l_mag     = self.mag_loss(est_spec, tgt_spec)
        l_mrsstft = self.mrsstft_loss(est_wav, tgt_wav)
        l_sisnr   = self.sisnr_loss(est_wav, tgt_wav)

        total = (self.lambda_mag     * l_mag
               + self.lambda_mrsstft * l_mrsstft
               + self.lambda_sisnr   * l_sisnr)
        return total, l_mag, l_mrsstft, l_sisnr

    @torch.no_grad()
    def debug_sisnr(self, est_spec, tgt_spec, out_dir="debug_sisnr", sr=16000, prefix="sample"):
        """
        Convert est_spec and tgt_spec back to waveforms, save them as .wav files,
        and print per-sample SI-SNR so you can listen and diagnose the loss value.

        Args:
            est_spec : (B, F, T, 2)  model output STFT
            tgt_spec : (B, F, T, 2)  ground-truth STFT
            out_dir  : directory to write wav files into
            sr       : sample rate (default 16000)
            prefix   : filename prefix (e.g. "epoch019_step4500")

        Saved files (per batch item i):
            {out_dir}/{prefix}_b{i:02d}_est.wav   ← model output waveform
            {out_dir}/{prefix}_b{i:02d}_tgt.wav   ← ground-truth waveform
            {out_dir}/{prefix}_b{i:02d}_info.txt  ← SI-SNR value + stats
        """
        os.makedirs(out_dir, exist_ok=True)

        n_fft      = self._n_fft
        hop_length = self._hop

        est_wav = istft(est_spec, n_fft, hop_length)   # (B, L)
        tgt_wav = istft(tgt_spec, n_fft, hop_length)   # (B, L)

        min_len = min(est_wav.shape[-1], tgt_wav.shape[-1])
        est_wav = est_wav[..., :min_len]
        tgt_wav = tgt_wav[..., :min_len]

        # Zero-mean (same as SISNRLoss.forward)
        est_zm = est_wav - est_wav.mean(dim=-1, keepdim=True)
        tgt_zm = tgt_wav - tgt_wav.mean(dim=-1, keepdim=True)

        dot          = torch.sum(est_zm * tgt_zm, dim=-1, keepdim=True)
        s_ref_energy = torch.sum(tgt_zm ** 2, dim=-1, keepdim=True) + 1e-8
        proj         = dot * tgt_zm / s_ref_energy
        noise        = est_zm - proj

        proj_power  = torch.sum(proj  ** 2, dim=-1).clamp(min=1e-8)
        noise_power = torch.sum(noise ** 2, dim=-1).clamp(min=1e-8)
        per_sample_sisnr = 10 * torch.log10(
            (proj_power / noise_power).clamp(min=1e-10, max=1e10)
        )  # (B,)

        batch_size = est_wav.shape[0]
        for i in range(batch_size):
            est_np = est_wav[i].cpu().float().numpy()
            tgt_np = tgt_wav[i].cpu().float().numpy()
            sisnr_val = per_sample_sisnr[i].item()

            est_path  = os.path.join(out_dir, f"{prefix}_b{i:02d}_est.wav")
            tgt_path  = os.path.join(out_dir, f"{prefix}_b{i:02d}_tgt.wav")
            info_path = os.path.join(out_dir, f"{prefix}_b{i:02d}_info.txt")

            sf.write(est_path, est_np, sr)
            sf.write(tgt_path, tgt_np, sr)

            est_rms  = float(est_np.std())
            tgt_rms  = float(tgt_np.std())
            est_peak = float(abs(est_np).max())
            tgt_peak = float(abs(tgt_np).max())

            info_lines = [
                f"SI-SNR        : {sisnr_val:.4f} dB",
                f"Est  RMS/peak : {est_rms:.6f} / {est_peak:.6f}",
                f"Tgt  RMS/peak : {tgt_rms:.6f} / {tgt_peak:.6f}",
                f"Waveform len  : {len(est_np)} samples  ({len(est_np)/sr:.3f} s @ {sr} Hz)",
                f"Est wav       : {est_path}",
                f"Tgt wav       : {tgt_path}",
            ]
            with open(info_path, "w") as f:
                f.write("\n".join(info_lines) + "\n")

            print(f"[debug_sisnr] [{prefix}] b{i:02d}  SI-SNR={sisnr_val:+.2f} dB  "
                  f"est_rms={est_rms:.4f}  tgt_rms={tgt_rms:.4f}")

        batch_mean = per_sample_sisnr.mean().item()
        print(f"[debug_sisnr] [{prefix}] batch mean SI-SNR = {batch_mean:+.4f} dB  "
              f"(saved {batch_size} pairs to {out_dir}/)")
        return per_sample_sisnr
