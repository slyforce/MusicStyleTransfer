from music_style_transfer.MIDIUtil.defaults import *
from music_style_transfer.MIDIUtil.Note import Note


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

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, key):
        return self.notes[key]

    def copy_metainformation(self):
        melody = Melody(key=self.key,
                        bpm=self.bpm,
                        resolution=self.resolution,
                        slices_per_quarter=self.slices_per_quarter,
                        description=self.description)
        return melody

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
            if all([len(note) == 0 for note in result[i].notes]):
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


def melody_from_sequence_of_pitches(pitches):
    melody = Melody()
    for pitch in pitches:
        melody.notes.append(set([Note(pitch)]))
    return melody