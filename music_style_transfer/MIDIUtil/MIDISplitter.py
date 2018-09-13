from .MIDIReader import MIDIReader
from .MelodyWriter import MelodyWriter
from .Melody import Melody

class MIDISplitter:
    def __init__(self):
        self.midi_reader = MIDIReader()
        self.melody_writer = MelodyWriter()

    def read_and_write(self,
                       input: str,
                       output_prefix: str):
        melodies = self.midi_reader.read_file(input)[0]

        for melody in melodies:
            output_file = self.append_melody_description(input,
                                                         output_prefix,
                                                         melody)

            self.melody_writer.write_to_file(output_file,
                                             melody)

    def append_melody_description(self,
                                  input: str,
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

    args = parser.parse_args()

    midi_splitter = MIDISplitter()

    input_folder = args.input
    output_folder = args.output

    for file_name in glob.glob(input_folder + "/*.mid"):
        midi_splitter.read_and_write(file_name, output_folder)
