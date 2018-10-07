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
net_arg.add_argument('--e-n-layers', type=int, default=1)
net_arg.add_argument('--e-rnn-hidden-dim', type=int, default=128)
net_arg.add_argument('--e-emb-hidden-dim', type=int, default=64)
net_arg.add_argument('--e-dropout', type=float, default=0.0)

net_arg.add_argument('--latent-dim', type=int, default=64)

net_arg.add_argument('--d-n-layers', type=int, default=1)
net_arg.add_argument('--d-rnn-hidden-dim', type=int, default=128)
net_arg.add_argument('--d-dropout', type=float, default=0.0)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch-size', type=int, default=1)
data_arg.add_argument('--max-seq-len', type=int, default=64)
data_arg.add_argument('--slices-per-quarter-note', type=float, default=64)
data_arg.add_argument('--data', type=str, default='data')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--epochs', type=int, default=5000)
train_arg.add_argument('--learning-rate', type=float, default=3e-4)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--optimizer-params', type=str, default='')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load-from', type=str, default='')
misc_arg.add_argument('--checkpoint-frequency', type=int, default=5000)
misc_arg.add_argument('--sampling-frequency', type=int, default=1000)
misc_arg.add_argument('--num-checkpoints-not-improved', type=int, default=10)
misc_arg.add_argument('--out-samples', type=str, default=None)
misc_arg.add_argument('--model-output', type=str, default='models')
misc_arg.add_argument('--gpu', action='store_true')
misc_arg.add_argument('--toy', action='store_true')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config
