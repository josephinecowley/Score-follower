from typing import List
from sharedtypes import NoteInfo
import mido
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt


def process_midi_to_note_info(midi_path: str) -> List[NoteInfo]:
    """
    Function to process a MIDI file into a notes vs time format
    """
    mid = mido.MidiFile(midi_path)
    ret = process_MidiFile(mid)

    return ret


def dict_to_frequency_list(chords: dict) -> list:
    sorted_time_keys = sorted(chords.keys(), reverse=False)
    score = [chords[key] for key in sorted_time_keys]
    return score


def notes_to_chords(notes: List[NoteInfo]) -> dict:
    """
    Returns a dictionary with keys as the onset times and a list of frequencies as the values (e.g. chords or individual notes)
    """
    # Create a dictionary to group notes into
    grouped_notes = {}
    for note_info in notes:
        note_start_key = note_info.note_start
        if note_start_key not in grouped_notes:
            grouped_notes[note_start_key] = []
        grouped_notes[note_start_key].append(
            440 * (2 ** ((note_info.midi_note_num - 69) / 12.0)))  # Here we have converted from midi to frequencies!
    return grouped_notes


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


def plot_piece(chords: dict, num_it: int):
    # Plot each set of values corresponding to time key for the first 50 items
    for i, (time, frequencies) in enumerate(chords.items()):
        plt.scatter([time] * len(frequencies), frequencies,
                    label=str(time), marker='x')
        if i == num_it:  # Stop after plotting the first 50 items
            break
    # Add labels and title
    plt.xlabel('Time')
    # Set y-axis to a logarithmic scale
    plt.yscale('log')
    plt.ylabel('Frequency')
    plt.title('Frequency vs Time')
    return
