import numpy as np
from GP_models.onset_detection import detected_samples
import matplotlib.pyplot as plt
import helper

from datetime import datetime

scale_dict = {13.5: [262], 15.0: [294], 49.0: [330], 76: [349], 97: [
    392], 287.89: [440], 9876: [494], 12987.4: [523]}  # Made up midi to chords file
sorted_keys = sorted(scale_dict.keys(), reverse=False)
scale = [scale_dict[key] for key in sorted_keys]

sample_data, sample_rate = detected_samples(
    '/Users/josephine/Documents/Engineering /Part IIB/Score alignment project/Score-follower/wav_files/scale.wav', sample_length=2000, offset=3000, show=False, delta=0.12, num_samples=None)

time_samples = np.linspace(
    0, len(sample_data[0])/sample_rate, len(sample_data[0]))
prob_matrix = np.zeros((len(scale), len(sample_data)))


def lml_table(sample_data, scale):
    for i, sample in enumerate(sample_data):
        for j in range(len(scale)):
            prob_matrix[j, i] = - \
                helper.stable_nlml(time_samples, sample, f=scale[j])
    return prob_matrix


prob_mat = lml_table(sample_data, scale)

plt.imshow(prob_mat, cmap='coolwarm', interpolation='nearest')
plt.title("Covariance Matrix Heatmap")
plt.colorbar()
plt.show()


def find_maximum_path(prob_matrix):
    num_rows, num_cols = prob_matrix.shape
    max_path = np.zeros(num_cols, dtype=int)

    # Initialization
    viterbi_matrix = np.copy(prob_matrix)
    backpointer_matrix = np.zeros_like(prob_matrix, dtype=int)

    for i in range(1, num_rows):
        for j in range(num_cols):
            prev_states = viterbi_matrix[i - 1, :]
            max_prev_state = np.argmax(prev_states)
            viterbi_matrix[i, j] += prev_states[max_prev_state]
            backpointer_matrix[i, j] = max_prev_state

    # Find the final state with the maximum probability
    max_final_state = np.argmax(viterbi_matrix[-1, :])

    # Backtrack to find the maximum path
    max_path[-1] = max_final_state
    for i in range(num_cols - 1, 0, -1):
        max_path[i - 1] = backpointer_matrix[i, max_path[i]]

    # Return the coordinates of the steps along the maximum path
    path_coordinates = np.column_stack((max_path, np.arange(num_cols)))

    return max_path, path_coordinates


max_path, path_coordinates = find_maximum_path(prob_mat)
print("Maximum path:", max_path)
print("Path coordinates:", path_coordinates)
