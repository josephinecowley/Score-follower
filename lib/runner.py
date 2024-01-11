import numpy as np
from .args import Arguments
from collections import defaultdict
from .eprint import eprint
from midi.midi import process_midi_to_note_info, notes_to_chords, dict_to_frequency_list
from GP_models.helper import SM_kernel


class Runner:
    def __init__(self, args: Arguments):
        """
        Precondition: assuming args.sanitize() was called.
        """
        self.args = args
        self.frame_duration = self.args.frame_length/self.args.sample_rate
        self.frame_times = np.linspace(
            0, self.frame_duration, self.args.frame_length)
        self.__log(f"Initiated with arguments:\n{args}")

    def start(self):
        self.__log(f"STARTING")

        self.__log(f"Begin: preprocess score")
        score = self.__preprocess_score()
        self.__log(f"End: preprocess score")

        self.__log(f"Begin: precalculate covariance matrices")
        cov_dict = self.__precalculate_cov(score[:50])
        self.__log(f"End: precalculate covariance matrices")

    def __preprocess_score(self) -> list:
        """
        Return list of states
        """
        args = self.args
        note_info = process_midi_to_note_info(args.score_midi_path)
        self.__log("Finished getting note info from score midi")

        dic = notes_to_chords(note_info)
        self.__log("Finished getting chords from note info")

        score = dict_to_frequency_list(dic)
        self.__log("Finished getting states from chords")

        return score

    def __precalculate_cov(self, score: list) -> dict:
        """
        Return dictionary of covariance functions to speed up score following
        """
        args = self.args
        cov_dict = {}

        for state in score:
            if str(state) not in cov_dict:
                cov_dict[str(state)] = SM_kernel(
                    self.frame_times, self.frame_times, M=args.M,  f=state, sigma_f=args.sigma_f, T=args.T, v=args.v) + args.sigma_n**2 * np.eye(args.frame_length)

        return cov_dict

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
