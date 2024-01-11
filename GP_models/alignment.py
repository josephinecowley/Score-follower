import numpy as np
import helper
from collections import defaultdict


def score_alignment(sample_data: list, score: list, time_samples: np.ndarray, n: int, cov_dict=None):
    """
    An off-line mode score aligner, which takes a list of sample_frames
    and a score (list of states)
    TODO currently we have a uniform prior
    TODO need to sort out how the arguments are not different to the inputs if using cov_dict
    """
    note_num, audio_num = 0, 0
    path = []
    for sample in sample_data[:-n]:
        probabilities = []
        num_lookahead = min(len(score) - note_num + 1, n)
        for i in range(num_lookahead):
            if cov_dict is None:
                probabilities.append(helper.stable_nlml(
                    time_samples=time_samples, sigma_f=0.1, Y=sample,  M=10, normalised=False, f=score[note_num+i]))
            else:
                probabilities.append(helper.stable_nlml(time_samples=time_samples, sigma_f=0.1,
                                     Y=sample,  M=10, normalised=False, f=score[note_num+i], cov_dict=cov_dict))

        priors = np.ones(num_lookahead)
        probabilities = np.array(probabilities)
        probabilities = probabilities * priors

        index = np.argmin(probabilities)

        note_num += index
        audio_num += 1
        path.append((note_num, audio_num))

    return path
