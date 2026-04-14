# DeepVQE
A PyTorch implementation of DeepVQE described in [DeepVQE: Real Time Deep Voice Quality Enhancement for Joint Acoustic Echo Cancellation, Noise Suppression and Dereverberation](https://arxiv.org/pdf/2306.03177.pdf).


All basic Modules comes from
(https://github.com/Xiaobin-Rong/deepvqe)

Change the arch and add the training method

USAGE.md for usage

## Train examples

### 1) AEC only
```bash
python train.py \
    --aec_root /home/cmy/cmy/AEC-Challenge/datasets/synthetic \
    --noise_dir /home/cmy/cmy/3D-Speaker/egs/3dspeaker/sv-eres2netv2/data/raw_data/musan \
    --rir_dir /home/cmy/cmy/AEC-Challenge/datasets/RIRs \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4
```

### 2) AEC + DNS (DNS on-the-fly mixing)
```bash
python train.py \
    --aec_root /home/cmy/cmy/AEC-Challenge/datasets/synthetic \
    --dns_root /home/cmy/cmy/DNS-Challenge/datasets/dns \
    --noise_dir /home/cmy/cmy/3D-Speaker/egs/3dspeaker/sv-eres2netv2/data/raw_data/musan /home/cmy/cmy/DNS-Challenge/datasets/dns/datasets.noise \
    --rir_dir /home/cmy/cmy/DNS-Challenge/datasets/dns_16k/datasets.impulse_responses \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --save_dir checkpoints \
    --resume /home/cmy/cmy-deepvqe/checkpoints/epoch_001.pt
```

### 3) DNS preprocessed paired dataset (clean/noisy)

`dataset.py` supports a preprocessed DNS mode when `--dns_root` contains both:

- `clean/` (for example: `clean_fileid_4955.wav`)
- `noisy/` (for example: `..._fileid_4955.wav`)

Files are paired by the `fileid_<N>` suffix in filename.

Example root:

```text
/home/cmy/cmy/DNS-Challenge/datasets/training_set/
  ├── clean/
  └── noisy/
```

Training command:

```bash
python train.py \
    --aec_root /home/cmy/cmy/AEC-Challenge/datasets/synthetic \
    --dns_root /home/cmy/cmy/DNS-Challenge/datasets/training_set \
    --noise_dir /home/cmy/cmy/3D-Speaker/egs/3dspeaker/sv-eres2netv2/data/raw_data/musan \
    --rir_dir /home/cmy/cmy/DNS-Challenge/datasets/dns_16k/datasets.impulse_responses \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --save_dir checkpoints
```

In this mode, DNS samples use paired `(noisy, clean)` directly (no on-the-fly DNS noise generation).



debug

python debug_sisnr.py \
    --ckpt /home/cmy/cmy-deepvqe/checkpoints/epoch_019.pt \
    --dns_root /home/cmy/cmy/DNS-Challenge/datasets/training_set \
    --noise_dir /home/cmy/cmy/3D-Speaker/egs/3dspeaker/sv-eres2netv2/data/raw_data/musan \
    --n_batches 3 \
    --batch_size 4 \
    --out_dir ./debug_sisnr_output


/home/cmy/cmy/DNS-Challenge/datasets/dns/datasets.dev_testset/datasets/dev_testset/ms_realrec_emotional_laptopmicrophone_A3U20M3KJ10B1A_Creakingchair_near_Surprised_fileid_5.wav

python inference.py  --mic /home/cmy/cmy/DNS-Challenge/datasets/dns/datasets.dev_testset/datasets/dev_testset/ms_realrec_emotional_laptopmicrophone_A3U20M3KJ10B1A_Creakingchair_near_Surprised_fileid_5.wav  --out enhanced.wav --ckpt /home/cmy/cmy-deepvqe/checkpoints/epoch_038.pt

