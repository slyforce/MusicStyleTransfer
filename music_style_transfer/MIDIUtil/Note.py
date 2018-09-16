from .defaults import *
import numpy as np
import midi


class Note:
    def __init__(self,
                 midi_pitch: int = None):

        if midi_pitch is None:
            # The pitch of the note
            self.pitch = PITCH_C
            # The octave of the note
            self.octave = 0
        else:
            self.set_from_midi_pitch(midi_pitch)

        # The length of the note
        self.length = DEF_TICK_STEP_SIZE

        # Whether the note is being played or articulated
        self.articulated = False

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

    def get_length(self):
        return self.length

    def get_pitch(self):
        return self.pitch

    def get_octave(self):
        return self.octave

    # TODO: remove this and let an Embedding class work this out
    def get_feature(self):
        result = np.array({self.pitch, self.octave})
        return result

    def __str__(self):
        if self.pitch == SILENCE:
            return "S_" + str(self.octave)
        else:
            return midi.NOTE_NAMES[self.pitch] + "_" + str(self.octave)

    def is_silence(self):
        return self.pitch == SILENCE


class SilenceNote(Note):
    def __init__(self):
        Note.__init__(self)
        self.octave = 0
        self.pitch = SILENCE
