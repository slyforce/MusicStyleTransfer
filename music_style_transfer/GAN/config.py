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
net_arg.add_argument('--z_dim', type=int, default=64)
net_arg.add_argument('--debug', type=str2bool, default=False)

#net_arg.add_argument('--D_h_dim', type=int, default=64, choices=[64, 128])

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument(
    '--dataset',
    type=str,
    default='/home/miguel/src/GeneticComposition/data/training_test')
data_arg.add_argument('--batch_size', type=int, default=1)
data_arg.add_argument('--height', type=int, default=50)
data_arg.add_argument('--width', type=int, default=50)
data_arg.add_argument('--channels', type=int, default=3)
data_arg.add_argument('--restrict', type=str, default='guitar')
data_arg.add_argument('--use_cgan', type=str2bool, default=True)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=500000)
train_arg.add_argument('--d_lr', type=float, default=0.00008)
train_arg.add_argument('--g_lr', type=float, default=0.00008)
train_arg.add_argument('--lr_lower_boundary', type=float, default=0.00002)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma', type=float, default=0.5)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=200)
misc_arg.add_argument('--save_step', type=int, default=5000)
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--sample_dir', type=str, default='samples')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--model_dir', type=str, default='models')
misc_arg.add_argument(
    '--test_data_path',
    type=str,
    default=None,
    help='directory with images which will be used in test sample generation')
misc_arg.add_argument(
    '--sample_per_image',
    type=int,
    default=64,
    help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)


def get_config():
    config, unparsed = parser.parse_known_args()
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed
