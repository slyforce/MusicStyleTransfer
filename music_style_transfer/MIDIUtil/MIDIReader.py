import midi

from .Melody import Melody
from .MelodyWriter import MelodyWriter
from .Note import Note
from .defaults import *


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

    def read_file(self, file_name):
        # Array of Melody objects
        result = []

        pattern = midi.read_midifile(file_name)

        resolution = pattern.resolution
        note_window = DEF_TICK_STEP_SIZE

        for idx, track in enumerate(pattern):
            new_melody = Melody()
            new_melody.set_tempo(resolution)

            notes_played = []
            track.make_ticks_abs()
            for event in track:
                if (isinstance(event, midi.NoteOnEvent)
                        or isinstance(event, midi.NoteOffEvent)):
                    if event.data[1] > 0:
                        # We got the duration of the pitch last played
                        tick_information = TickInformation()
                        tick_information.note = event.data[0]
                        tick_information.start_tick = event.tick

                    elif event.data[1] == 0:
                        # This gives us the duration of the pitch
                        tick_information.end_tick = event.tick
                        notes_played.append(tick_information)
                elif isinstance(event, midi.TrackNameEvent):
                    new_melody.description = event.text

            new_melody.notes = []
            # TODO: works in python 3.5?
            last_tick = reversed(track).next().tick
            for i in range(0, last_tick, int(note_window)):
                new_note = Note()
                junk_elements = []

                for tick in notes_played:
                    # We already handled this time point
                    if i >= tick.end_tick:
                        junk_elements.append(tick)

                    if tick.start_tick <= i and i < tick.end_tick:
                        # A note is being played in the time frame
                        new_note.set_from_midi_pitch(tick.note)
                        new_melody.notes.append(new_note)

                        # Do not repeat the note if it was already played once
                        # before
                        if tick.played:
                            new_note.articulated = True

                        tick.played = True
                        break
                    elif i < tick.start_tick:
                        # We have yet to check if there is a note or silence
                        new_note.pitch = SILENCE
                        new_melody.notes.append(new_note)
                        break

                # Reduce notes played container
                for el in junk_elements:
                    notes_played.remove(el)

            # Check if the track is too small
            # This can be the case for description tracks
            if len(new_melody.notes) < 10:
                #print "Too short track. Ignoring it."
                continue

            result.append(new_melody)

        result = self.clean_melodies(result)

        return result, note_window

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
