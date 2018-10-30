#!/bin/bash

source /home/slyforce/src/MusicStyleTransfer/venv/bin/activate;

python -m music_style_transfer.VarAutoEncoder.main \
--batch-size 128 \
--kl-loss 1.0 \
--out-samples /tmp/out \
--validation-split 0.1 \
--max-seq-len 32 \
--slices-per-quarter-note 4 \
--data ./work/data/guitar_bass \
--sampling-frequency 5000 \
--checkpoint-frequency 2000 \
--num-checkpoints-not-improved 32 \
--epochs 10000 \
--optimizer adam \
--optimizer-params clip_gradient:1.0 \
--learning-rate 0.0003 \
--model-output guitar_bass_model/ \
--positive-label-upscaling \
--label-smoothing 0.1 \
--e-n-layers 1 --e-dropout 0.2 --e-rnn-hidden-dim 300 --e-emb-hidden-dim 128 \
--latent-dim 128 \
--d-n-layers 1 --d-rnn-hidden-dim 300 --d-dropout 0.2 --gpu 

            

