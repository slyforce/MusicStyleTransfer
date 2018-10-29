from music_style_transfer.MIDIUtil.defaults import *
import numpy as np
import midi


class Note:
    def __init__(self,
                 midi_pitch: int = None,
                 articulated: bool = True):

        if midi_pitch is None:
            # The pitch of the note
            self.pitch = PITCH_C
            # The octave of the note
            self.octave = 0
        else:
            self.set_from_midi_pitch(midi_pitch)

        # Whether the note is being played or articulated
        self.articulated = articulated

    def set_from_midi_pitch(self, midi_pitch: int):
        if midi_pitch == 120:
            self.octave = 0
            self.pitch = SILENCE
        else:
            self.octave = midi_pitch // N_PITCHES
            self.pitch = midi_pitch % N_PITCHES

    def get_midi_index(self):
        if self.pitch == SILENCE:
            return N_PITCHES * N_OCTAVES
        else:
            return self.octave * N_PITCHES + self.pitch

    def __str__(self):
        if self.pitch == SILENCE:
            return "S_" + str(self.octave)
        else:
            return midi.NOTE_NAMES[self.pitch] + "_" + str(self.octave)

    def __cmp__(self, other):
        return self == other

    def __eq__(self, other):
        return self.articulated == other.articulated and self.pitch == other.pitch and self.octave == other.octave

    def __neq__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.pitch, self.octave, self.articulated))

    def __repr__(self):
        return self.__str__()

    def is_silence(self):
        return self.pitch == SILENCE


class SilenceNote(Note):
    def __init__(self):
        Note.__init__(self)
        self.octave = 0
        self.pitch = SILENCE
