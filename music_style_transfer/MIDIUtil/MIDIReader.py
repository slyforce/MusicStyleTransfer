import midi

from .Melody import Melody
from .Note import Note
from music_style_transfer.MIDIUtil.defaults import *

from typing import List


class TickInformation:
    '''
    Helper class to store which pitch is being played a tick time interval
    '''

    def __init__(self, notes=None):
        self.notes = notes
        if self.notes is None:
            self.notes = set()


class MIDIReader:
    def __init__(self,
                 slices_per_quarter_note: int):
        self.slices_per_quarter_note = slices_per_quarter_note

        print(
            "Time slices per quarter note: {}".format(
                self.slices_per_quarter_note))

    def _extract_bpm(self, pattern):
        """
        :param pattern: MIDI Pattern
        :return: First BPM found in file or if none is present the default BPM
        """
        for track in pattern:
            for event in track:
                if isinstance(event, midi.SetTempoEvent):
                    return event.get_bpm()
        return DEFAULT_BPM

    def read_file(self, file_name):
        # Array of Melody objects
        result = []

        pattern = midi.read_midifile(file_name)
        # resolution: ticks per quarter
        # bpm: beats per minute
        resolution = pattern.resolution
        bpm = self._extract_bpm(pattern)

        note_window = int(resolution // self.slices_per_quarter_note)

        #print("File resolution: {}".format(resolution))
        #print("BPM: {}".format(bpm))
        #print("Ticks per time-step: {}".format(note_window))

        for idx, track in enumerate(pattern):
            new_melody = Melody(
                bpm=bpm,
                resolution=resolution,
                slices_per_quarter=self.slices_per_quarter_note)
            new_melody.notes = self._parse_track(track, note_window)

            # Check if the track is too small
            # This can be the case for description tracks
            if len(new_melody) < 10:
                print('Warning: {} contains melodies of length {} < 10. Discarding'.format(file_name, len(new_melody.notes)))
                continue

            result.append(new_melody)

        assert len(result) > 0
        return result

    def _parse_track(self, track: midi.Track, tick_step: int):
        prev_t = 0
        open_notes = set()
        newly_open_notes = set()

        notes_played = []
        for event in track:
            cur_t = prev_t + event.tick

            self._write_pitches_being_played(cur_t, open_notes, newly_open_notes, prev_t, notes_played, tick_step)

            if (isinstance(event, midi.NoteOnEvent)
                    or isinstance(event, midi.NoteOffEvent)):
                [note, velocity] = event.data
                if velocity > 0:
                    # We got the duration of the pitch last played
                    newly_open_notes.add(Note(note, articulated=False))

                elif velocity == 0:
                    # This gives us the duration of the pitch
                    for open_note in open_notes:
                        if open_note.get_midi_index() == note:
                            open_notes.remove(open_note)
                            break
            else:
                continue

            prev_t = cur_t

        return notes_played

    def _write_pitches_being_played(self, cur_t, open_notes, newly_open_notes, prev_t, tick_infos, tick_step):
        start_tick = prev_t - (prev_t % tick_step)
        end_tick = cur_t - (cur_t % tick_step)

        for t in range(start_tick, end_tick, tick_step):

            notes_being_played = open_notes.copy()

            for note in newly_open_notes:
                notes_being_played.add(note)
                open_notes.add(Note(midi_pitch=note.get_midi_index(), articulated=True))

            newly_open_notes.clear()

            tick_infos.append(notes_being_played)

    def clean_melodies(self, melodies):
        result = []
        for melody in melodies:
            self.remove_silence(melody)
            result.append(melody)

        return result

    def remove_silence(self, melody):
        # todo: needs to be adapted to sets of notes
        raise NotImplemented
        junk_indices = []

        # Accumulate silence at the beginning of the melody
        for i, note in enumerate(melody.notes):
            if note.pitch == SILENCE:
                junk_indices.append(i)
            else:
                break

        # Accumulate silence at the end of the melody
        # TODO: Come up with a better list to iterate...
        for i, note in enumerate(list(reversed(melody.notes))):
            if note.pitch == SILENCE:
                junk_indices.append(i)
            else:
                break

        # Now remove all indices
        # Note that the list must be reversed to be able to pop correctly
        for i in reversed(junk_indices):
            melody.notes.pop(i)

        junk_indices = []

        # Eliminate too long sequences of silence that would not influence the
        # scoring
        counter = 0
        for i, note in enumerate(melody.notes):
            if note.pitch == SILENCE:
                counter += 1
            else:
                counter = 0

            if counter >= MAXIMUM_SEQUENCE_LENGTH:
                junk_indices.append(i)

        # Now remove all indices (again)
        # Note that the list must be reversed to be able to pop correctly
        for i in reversed(junk_indices):
            melody.notes.pop(i)

