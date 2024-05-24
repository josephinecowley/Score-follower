from ..eprint import eprint
import numpy as np
from GP_models.helper import stable_nlml
from lib.sharedtypes import (
    AudioFrameQueue,
    FollowerOutputQueue,
)


class Viterbi:
    def __init__(
        self,

        # Score and performance parameters
        follower_output_queue: FollowerOutputQueue,
        audio_frames_queue: AudioFrameQueue,
        score: list,
        time_to_next: list,
        score_times: np.ndarray,

        # Alignment parameters
        hop_length: int,
        window: int,
        threshold: int,
        state_duration_model: bool,
        scale_factor: float,

        # GP model hyperparameters
        sigma_f: float,
        sigma_n: float,
        cov_dict: dict,
        T: float,
        v: float,
        M: int,
        frame_length: int,
        frame_times: np.ndarray,
        sample_rate: float,

    ):

        self.follower_output_queue = follower_output_queue
        self.audio_frames_queue = audio_frames_queue
        self.score = score
        self.time_to_next = time_to_next
        self.score_times = score_times

        self.hop_length = hop_length
        self.window = window
        self.threshold = threshold
        self.state_duration_model = state_duration_model
        self.scale_factor = scale_factor

        self.cov_dict = cov_dict
        self.time_samples = frame_times
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.T = T
        self.v = v
        self.M = M
        self.frame_length = frame_length
        self.sample_rate = sample_rate

        self.max_s = 0
        self.d = 0
        self.frame_no = 0

        self.__log("Initialised successfully")

    def follow(self):
        """
        Performs score following using an on-line implementation of the Viterbi algorithm.
        Writes to self.follower_outpu_queue
        """
        gamma = np.full((len(self.score), 1000), -np.inf, 'd')
        i = 0
        chunk = 0
        step = self.window//3  # Threshold to trigger next chunk.
        conversion_rate = None
        counter = []  # This keeps track of state durations, d

        frame = self.get_next_frame(increment_d=False)
        if frame is None:
            self.follower_output_queue.put(None)
            return

        lml = -stable_nlml(self.time_samples, frame, M=self.M,
                           f=self.score[0], T=self.T, v=self.v, cov_dict=self.cov_dict)
        lml_scaled = np.sign(lml) * np.abs(lml)**self.scale_factor
        gamma[0, 0] = lml_scaled

        advance_transition = np.log(0.5)
        self_transition = np.log(0.5)

        while True:

            # Terminate if final state reached
            if self.max_s == len(self.score) - 1:
                self.follower_output_queue.put(None)
                return

            # Get next audio frame and increment i
            frame = self.get_next_frame()
            i += 1
            if frame is None:
                self.follower_output_queue.put(None)
                return

            # If we need to increase the tabulation matrix
            if i >= gamma.shape[1]:
                desired_len = int(i * 1.5)
                columns_to_add = desired_len - gamma.shape[1]

                new_columns = np.full(
                    (gamma.shape[0], columns_to_add), -np.inf, 'd')
                gamma = np.append(gamma, new_columns, axis=1)

            # Re-calculate transmission probabilities if taking into account state duration models
            if self.state_duration_model and conversion_rate:
                expected = conversion_rate * \
                    self.time_to_next[self.max_s] / 1000
                p = 1 / (expected + 1)
                q = 1 - p
                advance_transition = np.sum(
                    [q**z * p for z in range(1, self.d+1)])
                self_transition = np.log(1 - advance_transition)
                advance_transition = np.log(advance_transition)

            # Iterate through states in window
            k0_index = chunk * step
            for k in range(k0_index, k0_index + self.window):
                lml = -stable_nlml(self.time_samples, frame, M=self.M,
                                   f=self.score[k], T=self.T, v=self.v, cov_dict=self.cov_dict)
                lml_scaled = np.sign(lml) * np.abs(lml)**self.scale_factor
                print("k: ", k, "lml: ", lml, "lml_scaled: ", lml_scaled)

                same_state = lml_scaled + \
                    gamma[k, i-1] + self_transition
                advance_state = lml_scaled + \
                    gamma[k-1, i-1] + advance_transition
                gamma[k, i] = np.max(
                    [same_state, advance_state])

            # Determine most likely state
            print("max_s is ", gamma[k0_index:k0_index+self.window, i])
            new_s = np.argmax(gamma[:, i])

            # If required, update state duration d
            if self.state_duration_model:
                if new_s == self.max_s:
                    self.d += 1
                else:
                    counter.append(self.d)
                    conversion_rates = 1000 * np.array(counter) / np.array(
                        self.time_to_next[:len(counter)])
                    # We take the running mean average
                    conversion_rate = np.mean(conversion_rates)
                    self.d = 1

            self.max_s = new_s

            print(self.max_s, flush=True)
            self.follower_output_queue.put(
                (self.max_s, self.score_times[self.max_s]))  # Also returning score times for legacy reasons

            # Update chunk
            if self.max_s >= k0_index + self.window - step:
                chunk += 1

    def get_next_frame(self,  increment_d: bool = True):
        """
        Check the frame is above threshold value, if not continue to get frames.
        If frame is None, return None to output queue and terminate.
        """
        frame = self.audio_frames_queue.get()
        self.frame_no += 1
        sum = np.sum(np.array(frame, dtype=np.int64)**2)
        # sum = np.sum(np.array(frame)**2)  # for mode 2

        while sum < self.threshold:
            self.frame_no += 1

            if increment_d:
                self.d += 1

            self.__log(f"Amplitude too small, moving onto next audio frame")
            frame = self.audio_frames_queue.get()
            sum = np.sum(np.array(frame, dtype=np.int64)**2)
            # sum = np.sum(np.array(frame)**2)

        return frame

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
