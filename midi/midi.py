from typing import List
from sharedtypes import NoteInfo
import mido  # type: ignore
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt


def process_midi_to_note_info(midi_path: str) -> List[NoteInfo]:
    """
    https://github.com/flippy-fyp/flippy-quantitative-testbench/blob/main/midi.py
    """
    mid = mido.MidiFile(midi_path)
    ret = process_MidiFile(mid)

    return ret


def process_MidiFile(mid: mido.MidiFile) -> List[NoteInfo]:
    tempo = get_tempo(mid.tracks[0])
    track_midi_note_info_ticks: List[List[NoteInfo]] = [
        process_track(track, mid.ticks_per_beat, tempo) for track in mid.tracks
    ]
    # flatten
    ret: List[NoteInfo] = list(chain.from_iterable(track_midi_note_info_ticks))
    # sort
    ret.sort(key=lambda x: x.note_start)
    return ret


def get_tempo(meta_track: mido.midifiles.tracks.MidiTrack) -> int:
    for msg in list(meta_track):
        if hasattr(msg, "tempo"):
            return msg.tempo
    raise ValueError("Cannot get track tempo")


def process_track(
    track: mido.midifiles.tracks.MidiTrack, ticks_per_beat: int, tempo: int
) -> List[NoteInfo]:
    """
    Args:
        track: a mido.MidiTrack
        ticks_per_beat: integer of the set ticks per beat 
        tempo: integer of tempo in mido format
    Returns:
        list of NoteInfo clases, which contain midi_not_number and note_start times
    """
    ret: List[NoteInfo] = []
    curr_tick = 0
    for msg in track:
        curr_tick += msg.time
        if hasattr(msg, "velocity"):
            if msg.velocity > 0 and msg.type == "note_on":
                ret.append(
                    NoteInfo(
                        msg.note,
                        mido.tick2second(
                            curr_tick, ticks_per_beat, tempo) * 1000,
                    )
                )
    return ret


mid = mido.MidiFile(
    '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/midi_files/waldstein_1.midi')
tempo = get_tempo(mid.tracks[0])
track1 = mid.tracks[1]
track_midi_note_info_ticks = [
    process_track(track, mid.ticks_per_beat, tempo) for track in mid.tracks
]
notes = process_midi_to_note_info(
    '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/midi_files/waldstein_1.midi')
note_numbers = []
for i in notes:
    note_numbers.append(i.midi_note_num)
note_numbers = np.array(note_numbers)
print(type(note_numbers))
x = np.arange(len(note_numbers))
plt.plot(x, note_numbers, 'x')
plt.show()
# for track in track1:
#     print(track)
