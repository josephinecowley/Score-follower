from typing import Callable, Literal, NewType, Tuple, Optional, Union, Dict, List
from dataclasses import dataclass
import numpy as np
import multiprocessing as mp


@dataclass
class NoteInfo:
    midi_note_num: int  # MIDI note number
    note_start: float  # note start time (ms)
    note_end: float = None  # note end time (ms)
    note_duration: float = None  # Note duration

    def __eq__(self, other):
        if not isinstance(other, NoteInfo):
            return False

        return (
            self.midi_note_num == other.midi_note_num
            and self.note_start == other.note_start
            and self.note_end == other.note_end
            and self.note_duration == other.note_end - other.note_start
        )

    def __hash__(self):
        return hash((self.midi_note_num, self.note_start, self.note_end, self.note_duration))


AudioFrame = NewType("AudioFrame", np.ndarray)  # type: ignore
AudioFrameQueue = NewType(
    "AudioFrameQueue", "mp.Queue[Optional[AudioFrame]]"
)

FollowerOutputQueue = NewType(
    "FollowerOutputQueue", "mp.Queue[Optional[DTWPathElemType]]"
)
MultiprocessingConnection = NewType(
    "MultiprocessingConnection", "mp.connection.Connection"
)

Mode = Literal["basic", "oltw", "viterbi"]
