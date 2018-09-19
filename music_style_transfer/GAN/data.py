import numpy as np
import mxnet as mx

from ..MIDIUtil.defaults import *
from ..MIDIUtil.MIDIReader import MIDIReader
from ..MIDIUtil.RangeRestrictor import GuitarRangeRestrictor, BassRangeRestrictor, RangeRestrictor
from ..MIDIUtil.Melody import Melody

from typing import Dict, List

import glob
import os


class Loader:
    def __init__(self,
                 path: str,
                 max_sequence_length: int,
                 slices_per_quarter_note: int,
                 range_type: str = ''):
        self.path = path
        self.max_sequence_length = max_sequence_length
        self.slices_per_quarter_note = slices_per_quarter_note
        self.range_type = range_type
        assert range_type == '', "No support for restricted ranges yet"

        self._initialize_restrictor()
        self.midi_reader = MIDIReader(self.slices_per_quarter_note)
        self.melodies = self.read_melodies()

    def read_melodies(self):
        print("Reading from {}".format(self.path))
        melodies = {}
        directories = next(os.walk(self.path))[1]
        for directory in directories:
            melodies[directory] = []
            for n_files, fname in enumerate(glob.glob(self.path + '/' + directory + "/*.mid")):
                melody = self.midi_reader.read_file(fname)[0]
                melodies[directory] += melody.split_based_on_sequence_length(
                    self.max_sequence_length)

            print("Read {} files from {}".format(n_files, directory))
        return melodies

    def _initialize_restrictor(self):
        if self.range_type == 'guitar':
            self.restrictor = GuitarRangeRestrictor()
        elif self.range_type == 'bass':
            self.restrictor = BassRangeRestrictor()
        else:
            self.restrictor = RangeRestrictor()

    def _restrict_melodies(self):
        raise NotImplementedError

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


class Dataset:
    def __init__(self,
                 batch_size: int):
        self.batch_size = batch_size

    def num_classes(self):
        raise NotImplementedError

    def num_tokens(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class ToyData(Dataset):
    def __init__(self,
                 batch_size: int):
        super(ToyData, self).__init__(batch_size)
        self.batch_size = 2
        self.iter = mx.io.NDArrayIter({'data0': mx.nd.array([[1, 2, 3, 0, 0],
                                                              [2, 3, 4, 0, 0]]),
                                        'data1': mx.nd.array([1,2])},
                                         batch_size=self.batch_size, shuffle=False)
    def num_classes(self):
        return 3

    def num_tokens(self):
        return 5

    def __iter__(self):
        self.iter.reset()
        for batch in self.iter:
            yield batch


class MelodyDataset(Dataset):
    def __init__(self,
                 batch_size: int,
                 melodies: Dict[str, List[Melody]]):
        super().__init__(batch_size)
        self.mask_offset = 1
        self._initialize(melodies)
        self._log_dataset()

        # no need to keep these objects since we will only use the iterator
        del self.melodies

    def _initialize(self, melodies):
        # sort melodies by class
        self.melodies = dict(sorted(melodies.items(), key=lambda x: x[0]))
        self.n_classes = len(self.melodies)
        self.n_melodies = sum([len(m) for m in self.melodies.values()])
        max_seq_lens = []
        for melodies in self.melodies.values():
            max_seq_lens += [max([len(x.notes) for x in melodies])]
        self.max_sequence_length = max(max_seq_lens)
        self._get_class_arrays()
        self._get_token_arrays()
        self.iter = mx.io.NDArrayIter({'data0': self.tokens,
                                       'data1': self.classes},
                                      batch_size=self.batch_size, shuffle=True)

    def _log_dataset(self):
        print("Tokens dataset shape {}".format(self.tokens.shape))
        print("Classes dataset shape {}".format(self.classes.shape))
        for c, m in self.melodies.items():
            print("Class {} has {} melodies of maximum length {}".format(c, len(m), max([len(x.notes) for x in m])))

    def num_classes(self):
        return self.n_classes

    def num_tokens(self):
        return N_FEATURES_WITH_SILENCE + self.mask_offset

    def _get_class_arrays(self):
        arrays = [
            np.full(
                shape=(
                    len(melodies),
                ),
                fill_value=i) for i,
            melodies in enumerate(
                self.melodies.values())]
        arrays_concat = np.concatenate(arrays, axis=0)
        self.classes = mx.nd.array(arrays_concat)

    def _get_token_arrays(self):

        arrays = []
        for melodies in self.melodies.values():
            a = np.zeros((len(melodies), self.max_sequence_length))
            for i, melody in enumerate(melodies):
                # +1 due to masking
                a[i, :len(melody.notes)] = np.array(
                    [n.get_midi_index() + self.mask_offset for n in melody.notes])

            arrays.append(a)

        arrays_concat = np.concatenate(arrays, axis=0)
        self.tokens = mx.nd.array(arrays_concat)

    def __iter__(self):
        self.iter.reset()
        for batch in self.iter:
            yield batch

