from midi import process_midi_to_note_info, process_track, get_tempo
import mido
import matplotlib.pyplot as plt
import numpy as np

mid = mido.MidiFile(
    '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/midi_files/waldstein_1.midi')
tempo = get_tempo(mid.tracks[0])

# Print notes and times of Midi file
notes = process_midi_to_note_info(
    '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/midi_files/waldstein_1.midi')
note_numbers = []
print(notes[0])
for i in notes:
    note_numbers.append(i.midi_note_num)
note_numbers = np.array(note_numbers)
x = np.arange(len(note_numbers))
plt.scatter(x, note_numbers, marker='x')
plt.title("Midi file converted into values of score onset time and MIDI note numbers")
plt.show()
