import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt

frame_length = 2048
audio_stream = librosa.stream(
    "/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/GP_models/audio_processing/librosa experiments/experiment.wav",
    block_length=100,
    frame_length=2000,
    hop_length=2000,
    mono=True,
    fill_value=0,
    duration=5,
)
for audio_slice in audio_stream:
    print(audio_slice)
