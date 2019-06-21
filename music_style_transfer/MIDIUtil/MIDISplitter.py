from music_style_transfer.MIDIUtil.midi_io import MIDIReader
from music_style_transfer.MIDIUtil.MelodyWriter import MelodyWriter
from music_style_transfer.MIDIUtil.Melody import Melody


def read_and_write(reader: MIDIReader,
                   writer: MelodyWriter,
                   fname: str,
                   output_prefix: str):
    print("Reading file {}".format(fname))
    melodies = reader.read_file(fname)

    for melody in melodies:
        out_fname = append_melody_description(fname,
                                              output_prefix,
                                              melody)

        print(
            "Writing file {}  Length {}".format(
                out_fname, len(
                    melody.notes)))
        writer.write_to_file(out_fname, melody)


def append_melody_description(input: str,
                              prefix: str,
                              melody: Melody):
    res = prefix
    # Get the file name from a possible sequence of folders
    midi_file_name = input.split('/')[-1]
    midi_file_name = midi_file_name.split('.mid')[0]  # Remove the ".mid"

    res += midi_file_name + "_" + melody.description + ".mid"
    res = res.replace(' ', '-')  # Replace all whitespaces with hyphens
    return res


if __name__ == '__main__':
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str,
                        help="directory path with midi files to read from")
    parser.add_argument("output", type=str,
                        help="output path for the split midi files")
    parser.add_argument(
        "--slices-per-quarter",
        type=float,
        default=4,
        help="how many notes should be created for a quarter note. \n"
        " Default: 4 (corresponds to 16th notes")

    args = parser.parse_args()

    reader = MIDIReader(args.slices_per_quarter)
    writer = MelodyWriter()

    input_folder = args.input
    output_folder = args.output

    for file_name in glob.glob(input_folder + "/*.mid"):
        read_and_write(reader, writer, file_name, output_folder)
