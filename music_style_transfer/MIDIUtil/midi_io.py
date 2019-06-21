from MIDIUtil.Melody import EventInformation, NoteOnEvent, NoteOffEvent, TimeshiftEvent

from .Melody import Melody

from music_style_transfer.MIDIUtil.defaults import *
import midi


class MIDIReader():
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
        raise NotImplementedError


class EventBasedMIDIReader(MIDIReader):
    def __init__(self):
        super().__init__(0)
        self.event_factory = EventInformation()

    def read_file(self, file_name):
        # Array of Melody objects
        result = []

        pattern = midi.read_midifile(file_name)
        # resolution: ticks per beat (quarter)
        # bpm: beats per minute
        # duration of a tick in minutes: resolution * bpm
        # duration of a tick in miliseconds: resolution * bpm / 60 * 1000
        resolution = pattern.resolution
        bpm = self._extract_bpm(pattern)

        # print("File resolution: {}".format(resolution))
        # print("BPM: {}".format(bpm))
        # print("Ticks per time-step: {}".format(note_window))

        for idx, track in enumerate(pattern):
            new_melody = Melody(
                bpm=bpm,
                resolution=resolution,
                slices_per_quarter=self.slices_per_quarter_note)
            new_melody.notes = self._parse_track(track)

            # Check if the track is too small
            # This can be the case for description tracks
            if len(new_melody) < 10:
                print('Warning: {} contains melodies of length {} < 10. Discarding'.format(file_name,
                                                                                           len(new_melody.notes)))
                continue

            result.append(new_melody)

        assert len(result) > 0
        return result

    def _parse_track(self, track):
        events = []

        prev_t, cur_t = 0, 0
        for event in track:
            cur_t += event.tick

            delta_t = cur_t - prev_t

            if isinstance(event, midi.NoteOnEvent) or isinstance(event, midi.NoteOffEvent):
                [note, velocity] = event.data
                while delta_t > 0:
                    events.append(EventInformation.create_timeshift_event(delta_t % MAX_TICKS))
                    delta_t -= MAX_TICKS

                if velocity > 0:
                    events.append(EventInformation.create_note_on_event(note))

                elif velocity == 0:
                    events.append(EventInformation.create_note_off_event(note))

                prev_t = cur_t

        return events


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
        self._write_track(melody, track)
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        pattern.append(track)
        midi.write_midifile(file_name, pattern)

    def _write_track(self, melody, track):
        tick_delay = 0
        for event in melody:
            if isinstance(event, TimeshiftEvent):
                tick_delay += event.get_tick_delay()

            elif isinstance(event, NoteOnEvent) or isinstance(event, NoteOffEvent):
                track.append(event.get_midi_event(int(tick_delay)))
                tick_delay = 0

        return

    def _create_bpm_event(self, melody):
        bpm_event = midi.SetTempoEvent()
        bpm_event.set_bpm(melody.bpm)
        return bpm_event


