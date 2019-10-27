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
        return int(self.id - NOTE_ON_EVENTS[0])

    def get_midi_event(self, tick_delay: int):
        return midi.NoteOnEvent(pitch=self.shifted_id,
                                tick=tick_delay,
                                velocity=127)


class NoteOffEvent(Event):
    def __init__(self, id):
        super().__init__(id)

    @property
    def shifted_id(self):
        return int(self.id - NOTE_OFF_EVENTS[0])

    def get_midi_event(self, tick_delay: int):
        return midi.NoteOffEvent(pitch=self.shifted_id,
                                 tick=tick_delay)


class TimeshiftEvent(Event):
    def __init__(self, id):
        super().__init__(id)

    @property
    def shifted_id(self):
        return int(self.id - TIMESHIFT_EVENTS[0])

    def get_tick_delay(self):
        return self.shifted_id * NUM_TICKS_IN_A_BIN



def get_melody_from_ids(ids):
    melody = Melody()
    melody.notes = [create_event_from_id(id) for id in ids if id >= FEATURE_OFFSET]
    return melody


def create_event_from_id(id):
    event = None
    if id >= NUM_EVENTS or id < NOTE_ON_EVENTS[0]:
        raise ValueError("ID {} is not in range [{}, {}]".format(id,
                                                                 NOTE_ON_EVENTS[0],
                                                                 NUM_EVENTS))
    elif id >= TIMESHIFT_EVENTS[0]:
        event = TimeshiftEvent(id)
    elif id >= NOTE_OFF_EVENTS[0]:
        event = NoteOffEvent(id)
    elif id >= NOTE_ON_EVENTS[0]:
        event = NoteOnEvent(id)

    return event


def create_note_on_event(pitch: int):
    return NoteOnEvent(NOTE_ON_EVENTS[0] + pitch)


def create_note_off_event(pitch: int):
    return NoteOffEvent(NOTE_OFF_EVENTS[0] + pitch)


def create_timeshift_event(timeshift_ticks: int):
    assert MIN_TICKS <= timeshift_ticks < MAX_TICKS, \
        "Time shift must be between {} ticks and {} ticks. It is {}.".format(MIN_TICKS,
                                                                             MAX_TICKS,
                                                                             timeshift_ticks)

    binned_shift = int((timeshift_ticks - MIN_TICKS) / NUM_TICKS_IN_A_BIN)
    return TimeshiftEvent(TIMESHIFT_EVENTS[0] + binned_shift)

