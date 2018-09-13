import numpy as np

from ..MIDIUtil.defaults import *
from ..MIDIUtil.MIDIReader import MIDIReader

from ..MIDIUtil.RangeRestrictor import GuitarRangeRestrictor, BassRangeRestrictor, RangeRestrictor

import glob


class Loader:
    def __init__(self, path, range_type):
        self.path = path
        self.range_type = range_type

        self._initialize_restrictor()
        self.midi_reader = MIDIReader()
        self.read_melodies()

    def read_melodies(self):
        self.melodies = []
        print("Reading from: ", self.path)
        for file_name in glob.glob(self.path + "/*.mid"):
            print("Reading:", file_name)
            self.melodies += self.midi_reader.read_file(file_name)[0]

        self._restrict_melodies()

    def _initialize_restrictor(self):
        if self.range_type == 'guitar':
            self.restrictor = GuitarRangeRestrictor()
        elif self.range_type == 'bass':
            self.restrictor = BassRangeRestrictor()
        else:
            self.restrictor = RangeRestrictor()

    def _restrict_melodies(self):
        modified_melodies = []
        for melody in self.melodies:
            modified_melodies.append(self.restrictor.restrict(melody))

        self.melodies = modified_melodies

    def get_feature_length(self):
        return self.restrictor.get_range_length() + 1  # +1 to include silence


class FeatureManager:
    def __init__(self, data_information):
        self.maximum_sequence_length = MAXIMUM_SEQUENCE_LENGTH
        self._initialize_feature_information(data_information)

        self._index_in_epoch = 0
        self.data = None

    def _initialize_feature_information(self, data_information):
        self.feature_length = data_information.get_range_length() + \
            1  # +1 to include silence
        self.feature_begin = data_information._range_begin
        self.silence_idx = self.feature_length - 1

        print("feature length:", self.feature_length)
        print("feature begin:", self.feature_begin)
        print("silence index:", self.silence_idx)

    def get_silence_idx(self):
        return self.silence_idx

    def get_feature_begin(self):
        return self.feature_begin

    def get_feature_length(self):
        return self.feature_length

    def generate_data(self, melodies):
        # assure that every melody is limited to the maximum sequence length
        splitMelodies = []
        for melody in melodies:
            splitMelodies += melody.split_based_on_sequence_length(
                self.maximum_sequence_length)

        # build the data set
        # BoW-encoding
        data = np.zeros(
            (len(splitMelodies),
             self.maximum_sequence_length,
             self.feature_length,
             1))
        for melodyIdx, melody in enumerate(splitMelodies):
            for noteIdx, note in enumerate(melody.notes):
                if note.is_silence():
                    data[melodyIdx, noteIdx, self.silence_idx, 0] = 1
                else:
                    data[melodyIdx, noteIdx, note.get_midi_index() -
                         self.feature_begin, 0] = 1

        self.data = data

        print("shape of training data:", self.data.shape)
        return self.data

    def train_batch(self, n):
        if self._index_in_epoch + n >= self.get_number_samples():

            start1 = self._index_in_epoch
            end1 = self.get_number_samples()
            result1 = self.data[start1:end1, :, :]

            start2 = 0
            end2 = n - (self.get_number_samples() - self._index_in_epoch)
            result2 = self.data[start2:end2, :, :]

            self._index_in_epoch = end2
            result = np.concatenate((result1, result2), axis=0)
        else:
            start = self._index_in_epoch
            end = self._index_in_epoch + n
            self._index_in_epoch += n

            result = self.data[start:end, :, :]

        return result

    def get_number_samples(self):
        return self.data.shape[0]
