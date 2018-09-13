from .defaults import *


class Melody:
    def __init__(self):
        # Tempo of the melody
        self.tempo = DEFAULT_TEMPO

        # The key of the melody
        self.key = PITCH_C

        # The notes in the melody
        self.notes = []

        # General information to the track stored in the MIDI file
        self.description = ''

    def set_key(self, pitch):
        self.key = pitch

    def set_tempo(self, tempo):
        self.tempo = tempo

    def split_based_on_sequence_length(self, maxLength):
        result = []

        for i in range(0, len(self.notes)):
            # create new melody object
            if i % maxLength == 0:
                result.append(Melody())

            # append the current note to the most-recent melody
            result[-1].notes.append(self.notes[i])

        # all melodies should have at least one note
        assert(len(result) != 0)

        return result

    def articulate_notes(self):
        self.notes[0].articulated = False
        for i in range(1, len(self.notes)):
            prev_note = self.notes[i - 1]
            cur_note = self.notes[i]

            if prev_note.get_midi_index() == cur_note.get_midi_index():
                cur_note.articulated = True
            else:
                cur_note.articulated = False
