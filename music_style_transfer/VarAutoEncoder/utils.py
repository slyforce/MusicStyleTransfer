import pprint

from music_style_transfer.MIDIUtil.defaults import *

import matplotlib.pyplot as mpl
import mxnet as mx
import numpy as np
import os
import re

from MIDIUtil.Melody import Melody
import pickle


def get_latest_checkpoint_index(model_folder: str):
    checkpoint = -1
    for file in os.listdir(model_folder):
        match = re.search(r"params.(\d)+", file)
        if match is not None:
            checkpoint = max(int(match.group(1)), checkpoint)

    if checkpoint == -1:
        raise ValueError("No checkpoints found in {}".format(model_folder))

    return checkpoint


def save_model(model: mx.gluon.HybridBlock, output_path: str):
    model.save_parameters(output_path)


def save_object(object, output_path: str):
    with open(output_path, "wb") as file:
        pickle.dump(object, file)


def load_object(path: str):
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def load_model_parameters(model: mx.gluon.HybridBlock, path: str, context: mx.Context):
    model.load_parameters(path, ctx=context)


def create_directory_if_not_present(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def visualize_melody(melody: Melody, offset: int = -1):
    pixels = np.zeros((len(melody), N_FEATURES_WITHOUT_SILENCE))

    for i, notes in enumerate(melody):
       for note in notes:
           pixels[i, note.get_midi_index() + offset] = 1.

    pixels = np.transpose(pixels, axes=(1, 0))
    mpl.imshow(pixels, cmap='gray')
    mpl.show()


def log_config(config):
    pprint.pprint("Using configuration: ")
    pprint.pprint(config)


def log_model_variables(model):
    print("Model variables: ")
    pprint.pprint(model.collect_params())