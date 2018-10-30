from typing import List

from music_style_transfer.MIDIUtil.defaults import *

import matplotlib.pyplot as mpl
import mxnet as mx
import numpy as np
import os

from MIDIUtil.Melody import Melody
from MIDIUtil.Note import Note

def save_model(model: mx.gluon.HybridBlock, output_path: str):
    model.save_parameters(output_path)

def load_model_parameters(model: mx.gluon.HybridBlock, path: str, context: mx.Context):
    model.load_parameters(path, ctx=context)

def create_directory_if_not_present(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_melody(melody: Melody, output_path: str = None):
    pixels = np.zeros((len(melody), DEF_NUMBER_NOTES))

    for i, notes in enumerate(melody):
       for note in notes:
           pixels[i, note.get_midi_index()] = 1.

    pixels = np.transpose(pixels, axes=(1,0))
    mpl.imshow(pixels, cmap='gray')

    if output_path is None:
        mpl.show()
    else:
        raise NotImplementedError("Please pass None to visualize melody")







