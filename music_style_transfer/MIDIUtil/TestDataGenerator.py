from .Melody import Melody
from .Note import Note
from .MelodyWriter import MelodyWriter
from . defaults import DEF_NUMBER_NOTES

from music_style_transfer.VarAutoEncoder.utils import create_directory_if_not_present

import argparse
import copy
import numpy as np
import sys

def generate_melody(max_seq_len: int):
    melody = Melody()
    for i in range(max_seq_len):
        melody.notes.append(Note(midi_pitch=np.random.randint(0, DEF_NUMBER_NOTES)))
    return melody

def pitch_shift_melody(melody: Melody, shift: int):
    for note in melody.notes:
        note.set_from_midi_pitch((note.get_midi_index() + shift) % DEF_NUMBER_NOTES)
    return melody

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--n-shifts", type=int, default=10)
    parser.add_argument("--n-melodies", type=int, default=10)
    args = parser.parse_args()

    print("Outputting files to {}".format(args.output), file=sys.stderr)
    print("In total creating {} files from {} shifts".format(args.n_shifts * args.n_melodies,
                                                             args.n_shifts))
    create_directory_if_not_present(args.output)

    melody_writer = MelodyWriter()
    for melody_idx in range(args.n_melodies):
        original_melody = generate_melody(args.max_seq_len)
        for shift in range(args.n_shifts):
            melody = pitch_shift_melody(copy.copy(original_melody), shift)

            create_directory_if_not_present(args.output + '/shift-{}/'.format(shift))
            melody_writer.write_to_file(args.output + '/shift-{}/melody-{}.mid'.format(shift, melody_idx), melody)

if __name__ == '__main__':
    main()