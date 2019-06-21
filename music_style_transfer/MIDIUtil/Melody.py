import midi
from music_style_transfer.MIDIUtil.defaults import *
from music_style_transfer.MIDIUtil.defaults import MAX_TICKS, MIN_TICKS, NUM_TICKS_IN_A_BIN, NUM_BINS


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


class EventInformation:

    note_on_events = (0, 127)
    note_off_events = (note_on_events[1] + 1,
                       note_on_events[1] + 128)

    timeshift_events = (note_off_events[1] + 1,
                        note_off_events[1] + NUM_BINS)

    @staticmethod
    def create_note_on_event(pitch: int):
        return NoteOnEvent(EventInformation.note_on_events[0] + pitch)

    @staticmethod
    def create_note_off_event(pitch: int):
        return NoteOffEvent(EventInformation.note_off_events[0] + pitch)

    @staticmethod
    def create_timeshift_event(timeshift_ticks: int):
        assert MIN_TICKS <= timeshift_ticks < MAX_TICKS, \
            "Time shift must be between {} ticks and {} ticks. It is {}.".format(MIN_TICKS,
                                                                                 MAX_TICKS,
                                                                                 timeshift_ticks)


        binned_shift = (timeshift_ticks - MIN_TICKS) / NUM_TICKS_IN_A_BIN
        return TimeshiftEvent(EventInformation.timeshift_events[0] + binned_shift)

    @staticmethod
    def num_events():
        return EventInformation.timeshift_events[1]


class Event:
    def __init__(self, id):
        self.id = id

    @property
    def shifted_id(self):
        raise NotImplementedError

    def get_midi_event(self, tick_delay: int):
        raise NotImplementedError


class NoteOnEvent(Event):
    def __init__(self, id):
        super().__init__(id)

    @property
    def shifted_id(self):
        return self.id - EventInformation.note_on_events[0]

    def get_midi_event(self, tick_delay: int):
        return midi.NoteOnEvent(pitch=self.shifted_id,
                                tick=tick_delay,
                                velocity=127)


class NoteOffEvent(Event):
    def __init__(self, id):
        super().__init__(id)

    @property
    def shifted_id(self):
        return self.id - EventInformation.note_off_events[0]

    def get_midi_event(self, tick_delay: int):
        return midi.NoteOffEvent(pitch=self.shifted_id,
                                 tick=tick_delay)


class TimeshiftEvent(Event):
    def __init__(self, id):
        super().__init__(id)

    @property
    def shifted_id(self):
        return self.id - EventInformation.timeshift_events[0]

    def get_tick_delay(self):
        return self.shifted_id * NUM_TICKS_IN_A_BIN