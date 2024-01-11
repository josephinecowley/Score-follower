from .args import Arguments
from .eprint import eprint


class Runner:
    def __init__(self, args: Arguments):
        """
        Precondition: assuming args.sanitize() was called.
        """
        self.args = args
        self.__log(f"Initiated with arguments:\n{args}")

    def start(self):
        self.__log(f"STARTING")

    def __log(self, msg: str):
        eprint(f"[{self.__class__.__name__}] {msg}")
