# DeepVQE
A PyTorch implementation of DeepVQE described in [DeepVQE: Real Time Deep Voice Quality Enhancement for Joint Acoustic Echo Cancellation, Noise Suppression and Dereverberation](https://arxiv.org/pdf/2306.03177.pdf).


All basic Modules comes from
(https://github.com/Xiaobin-Rong/deepvqe)

Change the arch and add the training method

USAGE.md for usage

python train.py     --aec_root /home/cmy/cmy/AEC-Challenge/datasets/synthetic     --noise_dir /home/cmy/cmy/3D-Speaker/egs/3dspeaker/sv-eres2netv2/data/raw_data/musan --rir_dir /home/cmy/cmy/AEC-Challenge/datasets/RIRs    --epochs 100     --batch_size 8     --lr 1e-4


python train.py \
    --aec_root /home/cmy/cmy/AEC-Challenge/datasets/synthetic \
    --dns_root /home/cmy/cmy/DNS-Challenge/datasets/dns \
    --noise_dir /home/cmy/cmy/3D-Speaker/egs/3dspeaker/sv-eres2netv2/data/raw_data/musan  /home/cmy/cmy/DNS-Challenge/datasets/dns/datasets.noise\
    --rir_dir /home/cmy/cmy/DNS-Challenge/datasets/dns_16k/datasets.impulse_responses \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --save_dir checkpoints \
    --resume /home/cmy/cmy-deepvqe/checkpoints/epoch_001.pt