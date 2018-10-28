from music_style_transfer.MIDIUtil.defaults import *


class Melody:
    def __init__(self,
                 key=PITCH_C,
                 bpm=DEFAULT_BPM,
                 resolution=DEFAULT_RESOLUTION,
                 slices_per_quarter=4,
                 description: str = ''):
        self.key = key
        self.bpm = bpm
        self.resolution = resolution
        self.slices_per_quarter = int(slices_per_quarter)
        self.description = description
        self.notes = []

    def split_based_on_sequence_length(self, max_length):
        result = []

        for i in range(0, len(self.notes)):
            # create new melody object
            if i % max_length == 0:
                result.append(
                    Melody(
                        key=self.key,
                        bpm=self.bpm,
                        resolution=self.resolution,
                        slices_per_quarter=self.slices_per_quarter,
                        description=self.description))

            # append the current note to the most-recent melody
            result[-1].notes.append(self.notes[i])

        # at least one melody should have been produced
        assert(len(result) != 0)

        for i in reversed(range(len(result))):
            if all([note.is_silence() for note in result[i].notes]):
                # all notes are silence
                result.pop(i)

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
