import numpy as np
import mxnet as mx

from MIDIUtil.defaults import *
from MIDIUtil.MIDIReader import MIDIReader
from MIDIUtil.RangeRestrictor import GuitarRangeRestrictor, BassRangeRestrictor, RangeRestrictor
from MIDIUtil.Melody import Melody

from typing import Dict, List

import glob
import os


class Loader:
    def __init__(self,
                 path: str,
                 max_sequence_length: int,
                 slices_per_quarter_note: int,
                 range_type: str = '',
                 pattern_identifer = None):
        self.path = path
        self.max_sequence_length = max_sequence_length
        self.slices_per_quarter_note = slices_per_quarter_note
        self.range_type = range_type
        assert range_type == '', "No support for restricted ranges yet"
        self.pattern_identifier = pattern_identifer

        self._initialize_restrictor()
        self.midi_reader = MIDIReader(self.slices_per_quarter_note)
        self.melodies = self.read_melodies()

    def read_melodies(self):
        print("Reading from {}".format(self.path))
        melodies = {}
        directories = next(os.walk(self.path))[1]
        for directory in sorted(directories):
            melodies[directory] = []
            for n_files, fname in enumerate(glob.glob(self.path + '/' + directory + "/*.mid")):
                melody = self.midi_reader.read_file(fname)[0]

                if self.pattern_identifier is not None:
                    melodies[directory] += self.pattern_identifier.parse(melody)
                else:
                    melodies[directory] += melody.split_based_on_sequence_length(self.max_sequence_length)

            print("Read {} files from {}".format(n_files + 1, directory))
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
                 batch_size: int = 3):
        super(ToyData, self).__init__(batch_size)
        self.batch_size = batch_size
        self.iter = mx.io.NDArrayIter({'data0': mx.nd.array([[1, 2, 3, 0],
                                                             [2, 3, 4, 0],
                                                             [3, 4, 5, 0]]).one_hot(depth=self.num_tokens()),
                                       'data1': mx.nd.array([[1, 1, 1, 0],
                                                             [0, 1, 0, 0],
                                                             [1, 1, 0, 0]]).one_hot(depth=self.num_tokens()),
                                       'data2': mx.nd.array([0,1,2])},
                                        batch_size=self.batch_size, shuffle=False)
    def num_classes(self):
        return 3

    def num_tokens(self):
        return 6

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

        self._get_token_arrays()
        self._get_articulation_arrays()
        self._get_class_arrays()

        self.iter = mx.io.NDArrayIter({'data0': self.tokens,
                                       'data1': self.articulations,
                                       'data2': self.classes},
                                      batch_size=self.batch_size, shuffle=True)

    def _log_dataset(self):
        print("Tokens dataset shape {}".format(self.tokens.shape))
        print("Classes dataset shape {}".format(self.classes.shape))
        for c, m in self.melodies.items():
            print("Class {} has {} melodies of maximum length {}".format(c, len(m), max([len(x.notes) for x in m])))

    def num_classes(self):
        return self.n_classes

    def num_tokens(self):
        return N_FEATURES_WITHOUT_SILENCE + self.mask_offset

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

    def _get_articulation_arrays(self):
        articulation_arrays = []
        for melodies in self.melodies.values():
            articulations = np.zeros((len(melodies), self.max_sequence_length, self.num_tokens()))
            for i, melody in enumerate(melodies):
                for j, notes in enumerate(melody):
                    for note in notes:
                        articulations[i, j, note.get_midi_index() + self.mask_offset] = 1 if note.articulated else 0

            articulation_arrays.append(articulations)

        articulation_arrays_concat = np.concatenate(articulation_arrays, axis=0)

        self.articulations = mx.nd.array(articulation_arrays_concat)

    def _get_token_arrays(self):
        token_arrays = []
        for melodies in self.melodies.values():
            tokens = np.zeros((len(melodies), self.max_sequence_length, self.num_tokens()))

            # TODO: somehow optimize
            # this looks horribly inefficient...
            for i, melody in enumerate(melodies):
                for j, notes in enumerate(melody):
                    for note in notes:
                        tokens[i, j, note.get_midi_index() + self.mask_offset] = 1.

            token_arrays.append(tokens)

        token_arrays_concat = np.concatenate(token_arrays, axis=0)

        self.tokens = mx.nd.array(token_arrays_concat)

    def __iter__(self):
        self.iter.reset()
        for batch in self.iter:
            yield batch

def load_dataset(melodies: Dict[str, List[Melody]],
                 split_percentage: float,
                 batch_size: int):
    assert 0.0 <= split_percentage < 1.0

    if split_percentage == 0.0:
        # use the training data as validation data per default
        # todo: replace this with an optional validation path
        dataset = MelodyDataset(batch_size, melodies)
        return dataset, dataset

    train_melodies, valid_melodies = {}, {}
    for c, m in melodies.items():
        n_validation_melodies = int(split_percentage * len(m))
        valid_melodies[c] = m[:n_validation_melodies]
        train_melodies[c] = m[n_validation_melodies:]

    return MelodyDataset(batch_size, train_melodies),  MelodyDataset(batch_size, valid_melodies)




