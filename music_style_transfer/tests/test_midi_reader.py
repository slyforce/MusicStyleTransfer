import argparse
from MIDIUtil.midi_io import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='',
                        help='an integer for the accumulator')
    args = parser.parse_args()

    midi_reader = EventBasedMIDIReader(slices_per_quarter_note=4)
    midi_writer = MelodyWriter()
    melody = midi_reader.read_file(args.file)[0]
    midi_writer.write_to_file(args.file + '_rewrite.mid', melody)

    print("Output melody length {}".format(len(melody)))

if __name__ == '__main__':
    main()