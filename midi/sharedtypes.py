from dataclasses import dataclass


@dataclass
class NoteInfo:
    midi_note_num: int  # MIDI note number
    note_start: float  # note start time (ms)

    def __eq__(self, other):
        if not isinstance(other, NoteInfo):
            return False

        return (
            self.midi_note_num == other.midi_note_num
            and self.note_start == other.note_start
        )

    def __hash__(self):
        return hash((self.midi_note_num, self.note_start))
