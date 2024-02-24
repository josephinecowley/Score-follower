from ..eprint import eprint
from pydub import AudioSegment  # type: ignore
import time
from pydub.playback import play  # type: ignore


class Player:
    """
    Plays wave file
    """

    def __init__(self, wave_file_path: str, player_delay: float):
        self.wave = AudioSegment.from_wav(wave_file_path)
        self.player_delay = player_delay
        self.__log("Initialised successfully")

    def play(self):
        # TODO make this the correct amount of time and set this as an argument/or time constant
        time.sleep(self.player_delay)  # this is a compensation thing
        play(self.wave)

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
