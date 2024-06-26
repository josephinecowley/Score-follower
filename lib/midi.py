from typing import List
from lib.sharedtypes import NoteInfo
# from sharedtypes import NoteInfo
import mido
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def process_midi_to_note_info(midi_path: str) -> List[NoteInfo]:
    """
    Function to process a MIDI file into a notes vs time format.
    """

    mid = mido.MidiFile(midi_path)
    ret = process_MidiFile(mid)

    return ret


def dict_to_frequency_list(chords: dict) -> list:
    """
    Function to convert dictionary into a list of states.
    """

    sorted_time_keys = sorted(chords.keys(), reverse=False)
    time_to_next = [(sorted_time_keys[i+1] - sorted_time_keys[i])
                    for i in range(len(sorted_time_keys)-1)]
    score = [chords[key] for key in sorted_time_keys]
    score_times = np.cumsum(time_to_next)/1000
    score_times = np.insert(score_times, 0, 0)

    return score, time_to_next, score_times


def notes_to_chords(notes: List[NoteInfo], sustain: bool = False, remove_repeats: bool = False) -> dict:
    """
    Function that returns a dictionary with keys as the onset times and a list of
    frequencies as the values (e.g. chords or individual notes).
    """

    notes.sort(key=lambda x: x.note_start)
    grouped_notes = defaultdict(list)
    active_notes = set()

    for note_info in notes:
        note_frequency = round(
            440 * (2 ** ((note_info.midi_note_num - 69) / 12.0)))
        note_start_time = note_info.note_start
        note_end_time = note_info.note_end

        active_notes = {
            active_note for active_note in active_notes if active_note[0] >= note_start_time}

        active_notes.add((note_end_time, note_frequency))

        if sustain:
            if note_start_time not in grouped_notes:
                # We only ever need to add sustained notes to a group upon initialisation
                grouped_notes[note_start_time] = [active_note[1]
                                                  for active_note in active_notes]
            else:
                # We just need to add the current note (don't want to duplicate active notes)
                grouped_notes[note_start_time].append(note_frequency)
        else:
            if note_frequency not in grouped_notes[note_start_time]:
                grouped_notes[note_start_time].append(note_frequency)
    if remove_repeats:
        grouped_notes = remove_repeated_chords_from_dict(grouped_notes)

    return grouped_notes


def remove_repeated_chords_from_dict(grouped_notes: dict):
    """
    Function to remove repeated states/notes. 
    """

    keys_to_remove = []
    prev_value = None

    for key, value in grouped_notes.items():
        if value == prev_value:
            keys_to_remove.append(key)
        prev_value = value

    for key in keys_to_remove:
        del grouped_notes[key]

    return grouped_notes


def process_MidiFile(mid: mido.MidiFile) -> List[NoteInfo]:
    """
    Function which prepares the midi file, extracting note starts.
    """

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
        track: a mido.MidiTrack.
        ticks_per_beat: integer of the set ticks per beat. 
        tempo: integer of tempo in mido format. 

    Returns:
        List of NoteInfo clases, which contain midi_note_number, note_start times and note_end times.
    """

    ret: List[NoteInfo] = []
    active_notes = {}  # Dictionary to track active notes
    curr_tick = 0
    for msg in track:
        curr_tick += msg.time

        if hasattr(msg, "velocity"):
            if msg.velocity > 0 and msg.type == "note_on":
                # Note On message
                active_notes[msg.note] = curr_tick

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                # Note Off message or Note On with velocity 0 (simulated Note Off)
                if msg.note in active_notes:
                    start_time = mido.tick2second(
                        active_notes[msg.note], ticks_per_beat, tempo) * 1000
                    end_time = mido.tick2second(
                        curr_tick, ticks_per_beat, tempo) * 1000

                    ret.append(
                        NoteInfo(
                            msg.note,
                            start_time,
                            end_time,
                            (end_time - start_time),
                        )
                    )

                    # Remove the note from the active notes
                    del active_notes[msg.note]

    # Process remaining active notes without Note Off messages
    for note, start_tick in active_notes.items():
        start_time = mido.tick2second(start_tick, ticks_per_beat, tempo) * 1000
        end_time = None  # You can set this to some default value or leave it as None
        ret.append(
            NoteInfo(
                note,
                start_time,
                end_time
            )
        )

    return ret


def plot_piece(states: dict, num_it: int):
    """
    Helper function to plot a piece for visualisation of states.
    """

    for i, (time, frequencies) in enumerate(states.items()):
        plt.scatter([time] * len(frequencies), frequencies,
                    label=str(time), marker='x')
        if i == num_it:  # Stop after plotting the first 50 items
            break

    # Add labels and title
    plt.xlabel('Time')
    plt.yscale('log')
    plt.ylabel('Frequency')
    plt.title('Frequency vs Time')

    return
