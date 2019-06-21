#!/bin/bash

source /home/miguel/src/MusicStyleTransfer/venv/bin/activate;

python -m music_style_transfer.VarAutoEncoder.main \
--batch-size 4 \
--kl-loss 1.0 \
--out-samples /tmp/out \
--validation-split 0.2 \
--max-seq-len 32 \
--slices-per-quarter-note 4 \
--data ./work/data/guitar_bass \
--sampling-frequency 1000 \
--checkpoint-frequency 500 \
--num-checkpoints-not-improved 32 \
--epochs 10000 \
--optimizer adam \
--optimizer-params clip_gradient:1.0 \
--learning-rate 0.0003 \
--model-output guitar_bass_model/ \
--label-smoothing 0.0 \
--e-n-layers 1 \
--e-dropout 0.0 \
--e-rnn-hidden-dim 64 \
--e-emb-hidden-dim 64 \
--latent-dim 16 \
--d-n-layers 1 \
--d-rnn-hidden-dim 512 \
--d-dropout 0.0 
