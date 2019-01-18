from music_style_transfer.MIDIUtil.Melody import Melody, melody_from_sequence_of_pitches
from .defaults import SILENCE
import numpy as np

from collections import Counter, defaultdict

def get_pattern_identifier(type: str,
                           minimum_pattern_length: int,
                           maximum_pattern_length: int):
    if type == '':
        return None
    elif type == 'recurring':
        return RecurringPattern(minimum_pattern_length, maximum_pattern_length)
    else:
        raise NotImplementedError

class PatternIdentifierAbstract(object):
    def parse(self, melody: Melody):
        raise NotImplementedError

class RecurringPattern(PatternIdentifierAbstract):
    def __init__(self, minimum_pattern_length, maximum_pattern_length):
        """
        :param minimum_pattern_length:  Minimum time steps to be considered a pattern
        """
        assert minimum_pattern_length > 0
        assert maximum_pattern_length >= minimum_pattern_length
        self.minimum_pattern_length = minimum_pattern_length - 1
        self.maximum_pattern_length = maximum_pattern_length
        self.minimum_pattern_occurence = 2
        self.pattern_enumerator = defaultdict(int)

    def parse(self, melody: Melody):
        T = len(melody)
        hash_sequence = self._initialize_hash_sequence(melody)
        return self._fill_pattern_matrix(melody, hash_sequence)

    def _fill_pattern_matrix(self, original_melody, hash_sequence):
        melodies = []
        T = hash_sequence.shape[0]
        for t1 in range(T):
            for t2 in range(t1 + self.minimum_pattern_length, min(T, t1 + self.maximum_pattern_length)):
                cur_hash = hash(tuple(hash_sequence[t1:t2+1]))
                self.pattern_enumerator[cur_hash] += 1

                if self.pattern_enumerator[cur_hash] == self.minimum_pattern_occurence:
                    m = original_melody.copy_metainformation()
                    m.notes = [*original_melody[t1:t2+1]]
                    melodies.append(m)

        return melodies

    def _initialize_hash_sequence(self, melody):
        hash_sequence = np.zeros((len(melody),))
        for i, note in enumerate(melody):
            if len(note) == 0:
                hash_sequence[i] = hash(SILENCE)
            else:
                hash_sequence[i] = hash(*note)
        return hash_sequence

    def print_matrix(self, matrix):
        x1, x2 = matrix.shape
        for i in range(x1):
            for j in range(x2):
                print("{:.2f} ".format(matrix[i][j]), end='')
            print("")

def main():
    r = RecurringPattern(4)
    #test_sequence = melody_from_sequence_of_pitches([1,2])
    test_sequence = melody_from_sequence_of_pitches([1,2,3,4,120,120,1,2,3,4,1,2,3,4])

    out = r.parse(test_sequence)

    print("Original sentence {}".format([str(x) for x in test_sequence]))
    print("Read {} patterns".format(len(out)))
    for i in range(len(out)):
        print("Pattern {}: {}".format(i, [str(x) for x in out[i]]))

if __name__ == '__main__':
    main()
