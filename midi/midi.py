from typing import List
from sharedtypes import NoteInfo
import mido
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# import GP_models.helper


def process_midi_to_note_info(midi_path: str) -> List[NoteInfo]:
    """
    Function to process a MIDI file into a notes vs time format
    """
    mid = mido.MidiFile(midi_path)
    ret = process_MidiFile(mid)

    return ret


def dict_to_frequency_list(chords: dict) -> list:
    """
    TODO note: No longer deleting repeats here, instead in dictionary generation. This is only an option as well. 
    This was done so that using the score renderer it wouldn't create gaps. 
    Also should be ok once adding time information for the HMMs.
    """
    sorted_time_keys = sorted(chords.keys(), reverse=False)
    return [chords[key] for key in sorted_time_keys]
    # score_no_repeats = [score[0]]
    # for sublist in score[1:]:
    #     if sublist != score_no_repeats[-1]:
    #         score_no_repeats.append(sublist)
    # return score_no_repeats


def notes_to_chords(notes: List[NoteInfo], sustain: bool = False, remove_repeats: bool = False) -> dict:
    """
    Returns a dictionary with keys as the onset times and a list of frequencies as the values (e.g. chords or individual notes)
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
                # Else we just need to add the current note (don't want to duplicate active notes)
                grouped_notes[note_start_time].append(note_frequency)
        else:
            grouped_notes[note_start_time].append(note_frequency)
    if remove_repeats:
        grouped_notes = remove_repeated_chords_from_dict(grouped_notes)
    return grouped_notes


def remove_repeated_chords_from_dict(grouped_notes: dict):
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
