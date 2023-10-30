import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import helper

data, sr = librosa.load(
    '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/scale.wav')

onset_frames = librosa.onset.onset_detect(
    y=data, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
onset_times = librosa.frames_to_time(onset_frames)


print(onset_times)


wav_file = '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/scale.wav'
sample_rate, data = wav.read(wav_file)
onset_time_samples = onset_times * sample_rate

for i in range(len(onset_time_samples)):
    sample_data = data[int(onset_time_samples[i])
                           :int(onset_time_samples[i])+1000]
    helper.plot_fft(sample_data, sample_rate)
    plt.show()
