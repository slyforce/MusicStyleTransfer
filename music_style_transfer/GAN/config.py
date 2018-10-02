# -*- coding: utf-8 -*-
import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--g-n-layers', type=int, default=1)
net_arg.add_argument('--g-rnn-hidden-dim', type=int, default=128)
net_arg.add_argument('--g-emb-hidden-dim', type=int, default=64)
net_arg.add_argument('--noise-dim', type=int, default=64)

net_arg.add_argument('--d-n-layers', type=int, default=1)
net_arg.add_argument('--d-rnn-hidden-dim', type=int, default=128)
net_arg.add_argument('--d-emb-hidden-dim', type=int, default=64)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch-size', type=int, default=1)
data_arg.add_argument('--max-seq-len', type=int, default=64)
data_arg.add_argument('--slices-per-quarter-note', type=float, default=64)
data_arg.add_argument('--data', type=str, default='data')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--epochs', type=int, default=5000)
train_arg.add_argument('--g-learning-rate', type=float, default=3e-4)
train_arg.add_argument('--d-learning-rate', type=float, default=3e-4)
train_arg.add_argument('--discriminator-update-steps', type=int, default=1)
train_arg.add_argument('--label-smoothing', type=float, default=0.1)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load-from', type=str, default='')
misc_arg.add_argument('--checkpoint-frequency', type=int, default=5000)
misc_arg.add_argument('--sampling-frequency', type=int, default=1000)
misc_arg.add_argument('--out-samples', type=str, default=None)
misc_arg.add_argument('--model-output', type=str, default='models')
misc_arg.add_argument('--gpu', action='store_true')
misc_arg.add_argument('--toy', action='store_true')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config
