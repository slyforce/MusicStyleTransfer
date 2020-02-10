#!/bin/bash

source /home/miguel/src/MusicStyleTransfer/venv/bin/activate;

python -m music_style_transfer.VarAutoEncoder.main \
--batch-size 32 \
--kl-loss 1.0 \
--validation-split 0.0 \
--max-seq-len 64 \
--slices-per-quarter-note 4 \
--data ./work/data/guitar_bass \
--model-output models/guitar_bass \
--out-samples /tmp/out \
--sampling-frequency 2000 \
--checkpoint-frequency 1000 \
--num-checkpoints-not-improved 32 \
--epochs 10000 \
--optimizer adam \
--optimizer-params clip_gradient:1.0 \
--learning-rate 0.0003 \
--label-smoothing 0.0 \
--e-n-layers 2 \
--e-dropout 0.2 \
--e-rnn-hidden-dim 256 \
--e-emb-hidden-dim 256 \
--latent-dim 256 \
--d-n-layers 1 \
--d-rnn-hidden-dim 128 \
--d-dropout 0.2 
