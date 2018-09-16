#!/bin/bash

source /home/miguel/src/MusicStyleTransfer/venv/bin/activate;

python -m music_style_transfer.GAN.main \
--batch-size 2 \
--out-samples toietjwre \
--max-seq-len 4 \
--slices-per-quarter-note 4 \
--data /home/miguel/work/music_style_transfer/data/test_dataset \
--checkpoint-frequency 1000 \
--model-output test/ \
--g-n-layers 1 \
--g-rnn-hidden-dim 22 \
--g-emb-hidden-dim 22 \
--d-n-layers 1 \
--d-rnn-hidden-dim 22 \
--d-emb-hidden-dim 22 
