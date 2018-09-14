from .defaults import *
import midi


class MelodyWriter:
    def __init__(self):
        self.tempo = DEFAULT_TEMPO

    def get_midi_pitch(self, note):
        return note.octave * N_PITCHES + note.pitch

    def write_to_file(
            self,
            file_name,
            melody,
            tick_step_size=DEF_TICK_STEP_SIZE):
        pattern = midi.Pattern()
        pattern.resolution = melody.tempo

        track = midi.Track()

        notes = melody.notes

        next_tick_start = 0
        i = 0
        while i < len(notes):
            note = notes[i]

            if note.pitch != SILENCE and not note.articulated:
                pitch = self.get_midi_pitch(note)
                on = midi.NoteOnEvent(
                    tick=next_tick_start, velocity=127, pitch=pitch)
                track.append(on)

                # Count for the time-frame that the note was played
                next_tick_duration = tick_step_size

                # Count further articulations
                j = i + 1
                while j < len(notes) - 1 and notes[j].articulated:
                    next_tick_duration += tick_step_size
                    j += 1

                # Append the end of the note
                pitch = self.get_midi_pitch(note)
                off = midi.NoteOffEvent(tick=next_tick_duration, pitch=pitch)
                track.append(off)

                next_tick_start = 0

                # Set i to the new position
                i = j
                continue
            elif note.pitch == SILENCE:
                next_tick_start += tick_step_size

            i += 1

        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        pattern.append(track)
        midi.write_midifile(file_name, pattern)