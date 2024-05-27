from ..eprint import eprint
from pydub import AudioSegment
import time
from pydub.playback import play


class Player:
    """
    Plays .wav file. 
    """

    def __init__(self, wave_file_path: str, player_delay: float):
        self.wave = AudioSegment.from_wav(wave_file_path)
        self.player_delay = player_delay
        self.__log("Initialised successfully")

    def play(self):
        # This is added to align player and score follower.
        time.sleep(self.player_delay)
        play(self.wave)

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
