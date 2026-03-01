"""
DeepVQE training script.

Usage:
    python train.py --aec_root datasets/aec_challenge/synthetic \
                    --dns_root datasets/dns_challenge            \
                    --noise_dir datasets/noise                   \
                    --rir_dir datasets/rir                       \
                    --epochs 100 --batch_size 8 --lr 1e-3

Run `python train.py -h` for all options.
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from cmy_deepvqe import DeepVQE
from dataset import make_aec_dataset, make_dns_dataset, CombinedDataset
from loss import CombinedLoss
from utils import si_snr, istft


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Build dataloaders ────────────────────────────────────────────────────────

def build_datasets(cfg: Config):
    train_sets = []
    val_sets = []

    mix_kwargs = dict(
        sr=cfg.sr,
        segment_len=cfg.segment_len,
        n_fft=cfg.n_fft,
        hop=cfg.hop_length,
        snr_range=cfg.snr_range,
        ser_range=cfg.ser_range,
        gain_range_db=cfg.gain_range_db,
        rt60_range=cfg.rt60_range,
        echo_delay_range_ms=cfg.echo_delay_range_ms,
        echo_rt60_range=cfg.echo_rt60_range,
        reverb_prob=cfg.reverb_prob,
        echo_prob=cfg.echo_prob,
        noise_prob=cfg.noise_prob,
    )

    if cfg.aec_root is not None:
        train_sets.append(make_aec_dataset(
            aec_root=cfg.aec_root,
            noise_dir=cfg.noise_dir,
            rir_dir=cfg.rir_dir,
            use_pregenerated_echo=cfg.use_pregenerated_echo,
            **mix_kwargs,
        ))
    if cfg.dns_root is not None:
        train_sets.append(make_dns_dataset(
            dns_root=cfg.dns_root,
            noise_dir=cfg.noise_dir,
            rir_dir=cfg.rir_dir,
            **mix_kwargs,
        ))

    # Validation (disable augmentation: low reverb/echo variance)
    val_mix = {**mix_kwargs, "reverb_prob": 0.0, "echo_prob": 1.0, "noise_prob": 1.0}
    if cfg.val_aec_root is not None:
        val_sets.append(make_aec_dataset(
            aec_root=cfg.val_aec_root,
            noise_dir=cfg.val_noise_dir or cfg.noise_dir,
            rir_dir=cfg.val_rir_dir or cfg.rir_dir,
            use_pregenerated_echo=cfg.use_pregenerated_echo,
            **val_mix,
        ))
    if cfg.val_dns_root is not None:
        val_sets.append(make_dns_dataset(
            dns_root=cfg.val_dns_root,
            noise_dir=cfg.val_noise_dir or cfg.noise_dir,
            rir_dir=cfg.val_rir_dir or cfg.rir_dir,
            **val_mix,
        ))

    if not train_sets:
        raise ValueError("No training data specified. "
                         "Provide --aec_root and/or --dns_root.")

    train_ds = CombinedDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    val_ds = CombinedDataset(val_sets) if len(val_sets) > 1 else (val_sets[0] if val_sets else None)

    return train_ds, val_ds


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, cfg, epoch):
    model.train()
    device = cfg.device
    total_loss = 0.0
    total_spec = 0.0
    total_sisnr = 0.0
    n_steps = 0

    for step, (mic_spec, ref_spec, tgt_spec) in enumerate(loader):
        mic_spec = mic_spec.to(device)
        ref_spec = ref_spec.to(device)
        tgt_spec = tgt_spec.to(device)

        est_spec = model(mic_spec, ref_spec)

        loss, l_spec, l_sisnr = criterion(est_spec, tgt_spec)

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_spec += l_spec.item()
        total_sisnr += l_sisnr.item()
        n_steps += 1

        if (step + 1) % cfg.log_interval == 0:
            avg = total_loss / n_steps
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | "
                  f"Loss {avg:.4f}  Spec {total_spec/n_steps:.4f}  "
                  f"SI-SNR {total_sisnr/n_steps:.4f}")

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def validate(model, loader, criterion, cfg):
    model.eval()
    device = cfg.device
    total_loss = 0.0
    total_sisnr_metric = 0.0
    n_steps = 0

    for mic_spec, ref_spec, tgt_spec in loader:
        mic_spec = mic_spec.to(device)
        ref_spec = ref_spec.to(device)
        tgt_spec = tgt_spec.to(device)

        est_spec = model(mic_spec, ref_spec)
        loss, _, _ = criterion(est_spec, tgt_spec)
        total_loss += loss.item()

        # Time-domain SI-SNR metric
        est_wav = istft(est_spec, cfg.n_fft, cfg.hop_length)
        tgt_wav = istft(tgt_spec, cfg.n_fft, cfg.hop_length)
        min_len = min(est_wav.shape[-1], tgt_wav.shape[-1])
        metric = si_snr(est_wav[..., :min_len], tgt_wav[..., :min_len]).mean()
        total_sisnr_metric += metric.item()
        n_steps += 1

    avg_loss = total_loss / max(n_steps, 1)
    avg_sisnr = total_sisnr_metric / max(n_steps, 1)
    return avg_loss, avg_sisnr


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"Device: {device}")

    # Model
    model = DeepVQE(
        mic_channels=cfg.mic_channels,
        ref_channels=cfg.ref_channels,
        dec_channels=cfg.dec_channels,
        gru_hidden=cfg.gru_hidden,
        align_hidden=cfg.align_hidden,
        dmax=cfg.dmax,
        fe_compress=cfg.fe_compress,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f} M")

    # Loss
    criterion = CombinedLoss(
        compress=cfg.compress,
        alpha=cfg.loss_alpha,
        lambda_spec=cfg.lambda_spec,
        lambda_sisnr=cfg.lambda_sisnr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    ).to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
        )

    # Data
    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        )

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    if cfg.resume is not None and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {cfg.resume}, epoch {start_epoch}")

    # ── Training ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg, epoch)
        scheduler.step()

        msg = (f"Epoch {epoch}/{cfg.epochs} | "
               f"Train Loss {train_loss:.4f} | "
               f"LR {optimizer.param_groups[0]['lr']:.2e} | "
               f"Time {time.time()-t0:.1f}s")

        # Validation
        if val_loader is not None:
            val_loss, val_sisnr = validate(model, val_loader, criterion, cfg)
            msg += f" | Val Loss {val_loss:.4f} | Val SI-SNR {val_sisnr:.2f} dB"
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
        else:
            is_best = False

        print(msg)

        # Save checkpoint
        if epoch % cfg.save_interval == 0 or is_best:
            ckpt_path = os.path.join(cfg.save_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": train_loss,
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

            if is_best:
                best_path = os.path.join(cfg.save_dir, "best.pt")
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                }, best_path)
                print(f"  Saved best model -> {best_path}")

    print("Training complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepVQE")

    # data
    parser.add_argument("--aec_root", type=str, default=None,
                        help="AEC-Challenge synthetic root (nearend_mic_signal/ etc.)")
    parser.add_argument("--dns_root", type=str, default=None,
                        help="DNS-Challenge root (containing clean/)")
    parser.add_argument("--noise_dir", type=str, default=None,
                        help="Shared noise directory")
    parser.add_argument("--rir_dir", type=str, default=None,
                        help="Shared RIR directory (optional, synthetic if omitted)")
    parser.add_argument("--val_aec_root", type=str, default=None)
    parser.add_argument("--val_dns_root", type=str, default=None)
    parser.add_argument("--use_pregenerated_echo", action="store_true", default=True,
                        help="Use AEC echo_signal/ files directly (default: True)")
    parser.add_argument("--no_pregenerated_echo", dest="use_pregenerated_echo",
                        action="store_false",
                        help="Generate echo on-the-fly from far-end instead")

    # audio
    parser.add_argument("--segment_len", type=float, default=4.0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=256)

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "step"])
    parser.add_argument("--num_workers", type=int, default=4)

    # loss
    parser.add_argument("--lambda_spec", type=float, default=1.0)
    parser.add_argument("--lambda_sisnr", type=float, default=0.1)

    # checkpoint
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # device
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        aec_root=args.aec_root,
        dns_root=args.dns_root,
        noise_dir=args.noise_dir,
        rir_dir=args.rir_dir,
        val_aec_root=args.val_aec_root,
        val_dns_root=args.val_dns_root,
        use_pregenerated_echo=args.use_pregenerated_echo,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        segment_len=args.segment_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lr_scheduler=args.lr_scheduler,
        num_workers=args.num_workers,
        lambda_spec=args.lambda_spec,
        lambda_sisnr=args.lambda_sisnr,
        save_dir=args.save_dir,
        resume=args.resume,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
    )

    main(cfg)
