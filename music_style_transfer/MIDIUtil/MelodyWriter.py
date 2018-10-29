from music_style_transfer.MIDIUtil.defaults import *
import midi


class MelodyWriter:
    def __init__(self):
        self.tempo = DEFAULT_BPM

    def get_midi_pitch(self, note):
        return note.octave * N_PITCHES + note.pitch

    def write_to_file(
            self,
            file_name,
            melody):
        pattern = midi.Pattern()
        pattern.resolution = melody.resolution

        track = midi.Track()
        track.append(self._create_bpm_event(melody))

        self._write_track(
            melody, int(
                melody.resolution / melody.slices_per_quarter), track)

        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        pattern.append(track)
        midi.write_midifile(file_name, pattern)

    def _write_track(self, melody, tick_step_size, track):
        previous_pitches = set()

        tick_delay = 0
        for t, curr_notes in enumerate(melody):
            curr_pitches = [self.get_midi_pitch(x) for x in curr_notes]
            curr_articulation = [x.articulated for x in curr_notes]
            tick_delay += tick_step_size

            #print("{} {} {}".format(curr_pitches, curr_articulation, tick_step_size))
            on_events, off_events = [], []

            some_event = False
            # generate off events for all pitches that ended
            for i, pitch in enumerate(previous_pitches):
                if pitch not in curr_pitches:
                    off = midi.NoteOffEvent(tick=0 if some_event else tick_delay, pitch=pitch)
                    off_events.append(off)

                    #print("\toff event for pitch {} for {} ticks".format(pitch, 0 if some_event else tick_delay))
                    some_event = True
                    tick_delay = 0

            # now at the current notes and see which ones have started
            for pitch, articulation in zip(curr_pitches, curr_articulation):

                if not articulation:
                    # current pitch is being played

                    if pitch in previous_pitches:
                        # special case for when a note is played in succession
                        # generate an off event for the previous pitch
                        off = midi.NoteOffEvent(tick=0 if some_event else tick_delay, pitch=pitch)
                        off_events.append(off)
                        #print("\toff event for pitch {} for {} ticks".format(pitch, 0 if some_event else tick_delay))
                        tick_delay = 0

                    # generate an on event for the current pitch
                    # and keep track of it
                    on = midi.NoteOnEvent(tick=0 if some_event else tick_delay, velocity=127, pitch=pitch)
                    on_events.append(on)
                    #print("\ton event for pitch {} for {} ticks".format(pitch, 0 if some_event else tick_delay))

                    some_event = True
                    tick_delay = 0

            track.extend(off_events)
            track.extend(on_events)

            previous_pitches = curr_pitches

            if t > 200:
                break

    def _create_bpm_event(self, melody):
        bpm_event = midi.SetTempoEvent()
        bpm_event.set_bpm(melody.bpm)
        return bpm_event
