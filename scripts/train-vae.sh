#!/bin/bash

source /home/miguel/src/MusicStyleTransfer/venv/bin/activate;

python -m music_style_transfer.VarAutoEncoder.main \
--batch-size 2 \
--kl-loss 1.0 \
--validation-split 0.0 \
--max-seq-len 128 \
--slices-per-quarter-note 4 \
--data ./work/data/toy \
--model-output models/toy/ \
--out-samples /tmp/out \
--sampling-frequency 10 \
--checkpoint-frequency 500 \
--num-checkpoints-not-improved 32 \
--epochs 10000 \
--optimizer adam \
--optimizer-params clip_gradient:1.0 \
--learning-rate 0.001 \
--label-smoothing 0.0 \
--e-n-layers 1 \
--e-dropout 0.0 \
--e-rnn-hidden-dim 128 \
--e-emb-hidden-dim 128 \
--latent-dim 128 \
--d-n-layers 1 \
--d-rnn-hidden-dim 128 \
--d-dropout 0.0 
