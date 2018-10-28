from .Melody import Melody
from .Note import Note, SilenceNote
from music_style_transfer.MIDIUtil.defaults import *

class RangeRestrictor:
    def __init__(self):
        self._range_begin = 0
        self._range_end = N_FEATURES_WITHOUT_SILENCE

    def restrict(self, melody):
        '''
        Restrict the notes in a melody to be between an implemented range.
        Notes out of this range are mapped to silence.
        :param melody:
          A melody object
        :return:
          A melody object with the replaced notes
        '''

        new_melody = Melody()

        for i, note in enumerate(melody.notes):
            if not self.in_range(note.get_midi_index()):
                new_melody.notes.append(SilenceNote())
            else:
                new_melody.notes.append(note)

        return new_melody

    def get_range_length(self):
        return self._range_end - self._range_begin + 1

    def in_range(self, pitch):
        if self._range_begin <= pitch and pitch <= self._range_end:
            return True

        return False


class GuitarRangeRestrictor(RangeRestrictor):
    def __init__(self):
        super().__init__()
        self._range_begin = MIDI_GUITAR_BEGIN
        self._range_end = MIDI_GUITAR_END


class BassRangeRestrictor(RangeRestrictor):
    def __init__(self):
        super().__init__()
        self._range_begin = MIDI_BASS_BEGIN
        self._range_end = MIDI_BASS_END


if __name__ == '__main__':
    melody = Melody()
    melody.notes = [Note(pitch) for pitch in [10, 20, 0, 40, 50, 100, 150]]

    g_restrictor = GuitarRangeRestrictor()
    g_melody = g_restrictor.restrict(melody)

    print("Old melody:", [str(note) for note in melody.notes])
    print("New melody:", [str(note) for note in g_melody.notes])
