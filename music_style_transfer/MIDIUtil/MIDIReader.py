import midi

from .Melody import Melody
from .Note import Note
from .defaults import *

from typing import List


class TickInformation:
    '''
    Helper class to store which pitch is being played a tick time interval
    '''

    def __init__(self, note=SILENCE, start_tick=0, end_tick=0):
        self.note = note
        self.start_tick = start_tick
        self.end_tick = end_tick
        self.played = False


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
            notes_played, description = self._parse_track(track)
            new_melody.description = description
            self._generate_notes(new_melody, notes_played,
                                 note_window, track[-1].tick)

            # Check if the track is too small
            # This can be the case for description tracks
            if len(new_melody.notes) < 10:
                continue

            result.append(new_melody)

        result = self.clean_melodies(result)

        return result

    def _generate_notes(self,
                        melody: Melody,
                        notes_played: List[TickInformation],
                        note_window: int,
                        end_tick: int):
        melody.notes = []
        for t in range(0, end_tick, note_window):
            new_note = Note()
            junk_elements = []

            for note in notes_played:
                # We already handled this time point
                if t >= note.end_tick:
                    junk_elements.append(note)

                if note.start_tick <= t and t < note.end_tick:
                    # A note is being played in the time frame
                    new_note.set_from_midi_pitch(note.note)
                    melody.notes.append(new_note)

                    # Do not repeat the note if it was already played once
                    # before
                    if note.played:
                        new_note.articulated = True

                    note.played = True
                    break
                elif t < note.start_tick:
                    # We have yet to check if there is a note or silence
                    new_note.pitch = SILENCE
                    melody.notes.append(new_note)
                    break

            # Reduce notes played container
            for el in junk_elements:
                notes_played.remove(el)

    def _parse_track(self, track: midi.Track):
        notes_played = []
        description = ''
        track.make_ticks_abs()
        for event in track:
            if (isinstance(event, midi.NoteOnEvent)
                    or isinstance(event, midi.NoteOffEvent)):
                velocity = event.data[1]
                if velocity > 0:
                    # We got the duration of the pitch last played
                    tick_information = TickInformation()
                    tick_information.note = event.data[0]
                    tick_information.start_tick = event.tick

                elif velocity == 0:
                    # This gives us the duration of the pitch
                    tick_information.end_tick = event.tick
                    notes_played.append(tick_information)
            elif isinstance(event, midi.TrackNameEvent):
                description = event.text

        return notes_played, description

    def clean_melodies(self, melodies):
        result = []
        for melody in melodies:
            self.remove_silence(melody)
            result.append(melody)

        return result

    def remove_silence(self, melody):
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
