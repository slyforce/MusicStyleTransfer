import numpy as np
import mxnet as mx

from MIDIUtil.defaults import *
from MIDIUtil.midi_io import EventBasedMIDIReader
from MIDIUtil.Melody import Melody

from typing import Dict, List

import glob
import os


class Loader:
    def __init__(self,
                 path: str,
                 max_sequence_length: int,
                 slices_per_quarter_note: int):
        self.path = path
        self.max_sequence_length = max_sequence_length
        self.slices_per_quarter_note = slices_per_quarter_note

        self.midi_reader = EventBasedMIDIReader()
        self.melodies = self.read_melodies()

    def read_melodies(self):
        print("Reading from {}".format(self.path))
        melodies = {}
        directories = next(os.walk(self.path))[1]
        for directory in sorted(directories):

            melodies[directory] = []
            n_files = 0
            for n_files, fname in enumerate(glob.glob(self.path + '/' + directory + "/*.mid")):
                melody = self.midi_reader.read_file(fname)[0]
                melodies[directory].append(melody)

            print("Read {} files from {}".format(n_files + 1, directory))
        return melodies


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
        self.iter = mx.io.NDArrayIter({'data0': mx.nd.array([[1, 1, 2, 3, 0],
                                                             [1, 2, 3, 4, 0],
                                                             [1, 3, 4, 5, 0]]),
                                       'data1': mx.nd.array([4,4,4]),
                                       'data2': mx.nd.array([0,1,2])},
                                      label={'labels': mx.nd.array([[1, 2, 3, 0, 0],
                                                                    [2, 3, 4, 0, 0],
                                                                    [3, 4, 5, 0, 0]])},
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
                 maximum_sequence_length: int,
                 melodies: Dict[str, List[Melody]]):
        super().__init__(batch_size)
        self.max_seq_len = maximum_sequence_length + 1 # to accomodate eos and sos symbols
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
        self.seen_max_sequence_length = max(max_seq_lens)

        self._get_token_arrays()

        # prefix with numerals to ensure an ordering
        self.iter = mx.io.NDArrayIter({'0_tokens': self.tokens,
                                       '1_classes': self.classes},
                                      {'labels': self.labels},
                                      batch_size=self.batch_size, shuffle=True)

    def _log_dataset(self):
        print("Tokens dataset shape {}".format(self.tokens.shape))
        print("Classes dataset shape {}".format(self.classes.shape))
        for c, m in self.melodies.items():
            print("Class {} has {} melodies of maximum length {}".format(c, len(m), self.seen_max_sequence_length ))

    def num_classes(self):
        return self.n_classes

    def num_tokens(self):
        return N_FEATURES_WITHOUT_SILENCE + self.mask_offset

    def _get_token_arrays(self):
        all_tokens, all_labels, all_classes = [], [], []
        tokens, labels = None, None

        print(self.max_seq_len)
        for class_idx, melodies_for_class in enumerate(self.melodies.values()):

            for i, melody in enumerate(melodies_for_class):

                tokens = np.full((self.max_seq_len,), fill_value=PAD_ID)
                labels = np.zeros_like(tokens)
                tokens[0] = SOS_ID

                for j, event in enumerate(melody):
                    rel_index = j % (self.max_seq_len-1)
                    labels[rel_index] = tokens[rel_index+1] = FEATURE_OFFSET + event.id

                    if j % (self.max_seq_len - 1) == 0:
                        labels[-1] = EOS_ID
                        all_tokens.append(tokens)
                        all_labels.append(labels)
                        all_classes.append(class_idx)

                        tokens = np.zeros((self.max_seq_len, ))
                        labels = np.zeros_like(tokens)
                        tokens[0] = SOS_ID

                labels[rel_index + 1] = EOS_ID
                all_tokens.append(tokens)
                all_labels.append(labels)
                all_classes.append(class_idx)

            # possibly empty sequences if max sequence length splits input exactly
            if tokens.max() > 0.:
                all_tokens.append(tokens)
                all_labels.append(labels)
                all_classes.append(class_idx)

        self.tokens = mx.nd.array(np.concatenate(np.expand_dims(all_tokens, axis=0), axis=0))
        self.labels = mx.nd.array(np.concatenate(np.expand_dims(all_tokens, axis=0), axis=0))
        self.classes = mx.nd.array(all_classes)

        print("Tokens.shape {}".format(self.tokens.shape))
        print("Labels.shape {}".format(self.labels.shape))
        print("classes.shape {}".format(self.classes.shape))

        assert self.classes.size > 0, "Empty sequences were found"

    def __iter__(self):
        self.iter.reset()
        for batch in self.iter:

            # estimate the sequence length of the batch
            tokens = batch.data[0]
            seq_lens = mx.nd.where(tokens != PAD_ID,
                                   mx.nd.ones_like(tokens),
                                   mx.nd.zeros_like(tokens)).sum(axis=1)

            # insert it at the second position
            batch.data.insert(1, seq_lens)
            yield batch


def load_dataset(loader_train: Loader,
                 batch_size: int,
                 split_percentage: float = None,
                 loader_val: Loader = None):

    if loader_val is not None:
        train = MelodyDataset(batch_size, loader_train.max_sequence_length, loader_train.melodies)
        val = MelodyDataset(batch_size, loader_val.max_sequence_length, loader_val.melodies)
        return train, val

    if split_percentage is None:
        dataset = MelodyDataset(batch_size, loader_train.max_sequence_length, loader_train.melodies)
        return dataset, None

    assert 0.0 < split_percentage < 1.0

    train_split, valid_split = {}, {}
    for c, m in loader_train.melodies.items():
        n_validation_melodies = int(split_percentage * len(m))
        valid_split[c] = m[:n_validation_melodies]
        train_split[c] = m[n_validation_melodies:]

    return MelodyDataset(batch_size, loader_train.max_sequence_length, train_split),  MelodyDataset(batch_size, loader_train.max_sequence_length, valid_split)




