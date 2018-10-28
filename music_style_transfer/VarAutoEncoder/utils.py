from typing import List

import mxnet as mx
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

def construct_melody_from_integers(notes: List[int],
                                   articulations: List[bool],
                                   mask_offset: int=1):
    """
    Create a melody from a list of integers corresponding to midi pitches.
    """
    melody = Melody()
    melody.notes = [Note(midi_pitch=max(pitch-mask_offset, 0), articulated=articulation) for pitch, articulation in zip(notes, articulations)]
    return melody