"""
Loss functions for DeepVQE training.

Combines:
  - Complex compressed MSE on spectrogram (magnitude + phase aware)
  - Time-domain SI-SNR loss
"""

import torch
import torch.nn as nn

from utils import istft


class CompressedSpecLoss(nn.Module):
    """
    Compressed spectral loss in the STFT domain.
    L = alpha * ||  |Y|^c * e^{j*angle(Y)} - |S|^c * e^{j*angle(S)} ||^2
      + (1-alpha) * || |Y|^c - |S|^c ||^2

    where c is the compression factor.
    Operates on (B, F, T, 2) real-valued STFT tensors.
    """

    def __init__(self, compress=0.3, alpha=0.7):
        super().__init__()
        self.c = compress
        self.alpha = alpha

    def forward(self, est_spec, tgt_spec):
        """
        est_spec, tgt_spec: (B, F, T, 2)
        """
        est_real, est_imag = est_spec[..., 0], est_spec[..., 1]
        tgt_real, tgt_imag = tgt_spec[..., 0], tgt_spec[..., 1]

        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2 + 1e-12)
        tgt_mag = torch.sqrt(tgt_real ** 2 + tgt_imag ** 2 + 1e-12)

        # Magnitude loss
        mag_loss = torch.mean((est_mag.pow(self.c) - tgt_mag.pow(self.c)) ** 2)

        # Complex compressed loss
        est_compressed = torch.stack([
            est_mag.pow(self.c - 1) * est_real,
            est_mag.pow(self.c - 1) * est_imag
        ], dim=-1)
        tgt_compressed = torch.stack([
            tgt_mag.pow(self.c - 1) * tgt_real,
            tgt_mag.pow(self.c - 1) * tgt_imag
        ], dim=-1)
        complex_loss = torch.mean((est_compressed - tgt_compressed) ** 2)

        return self.alpha * complex_loss + (1 - self.alpha) * mag_loss


class SISNRLoss(nn.Module):
    """Negative SI-SNR loss in time domain."""

    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, est_spec, tgt_spec):
        """
        est_spec, tgt_spec: (B, F, T, 2) complex STFT
        Returns negative SI-SNR (to minimize).
        """
        est_wav = istft(est_spec, self.n_fft, self.hop_length)
        tgt_wav = istft(tgt_spec, self.n_fft, self.hop_length)

        # Align lengths
        min_len = min(est_wav.shape[-1], tgt_wav.shape[-1])
        est_wav = est_wav[..., :min_len]
        tgt_wav = tgt_wav[..., :min_len]

        # Zero-mean
        est_wav = est_wav - est_wav.mean(dim=-1, keepdim=True)
        tgt_wav = tgt_wav - tgt_wav.mean(dim=-1, keepdim=True)

        dot = torch.sum(est_wav * tgt_wav, dim=-1, keepdim=True)
        s_ref_energy = torch.sum(tgt_wav ** 2, dim=-1, keepdim=True) + 1e-8
        proj = dot * tgt_wav / s_ref_energy

        noise = est_wav - proj
        si_snr = 10 * torch.log10(
            torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8)
        )
        return -si_snr.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss = lambda_spec * CompressedSpecLoss + lambda_sisnr * SISNRLoss
    """

    def __init__(self, compress=0.3, alpha=0.7, lambda_spec=1.0, lambda_sisnr=0.1,
                 n_fft=512, hop_length=256):
        super().__init__()
        self.spec_loss = CompressedSpecLoss(compress=compress, alpha=alpha)
        self.sisnr_loss = SISNRLoss(n_fft=n_fft, hop_length=hop_length)
        self.lambda_spec = lambda_spec
        self.lambda_sisnr = lambda_sisnr

    def forward(self, est_spec, tgt_spec):
        l_spec = self.spec_loss(est_spec, tgt_spec)
        l_sisnr = self.sisnr_loss(est_spec, tgt_spec)
        return self.lambda_spec * l_spec + self.lambda_sisnr * l_sisnr, l_spec, l_sisnr
