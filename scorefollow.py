from lib.args import Arguments
from lib.runner import Runner
import argparse

if __name__ == "__main__":
    args = Arguments().parse_args()
    args.sanitise()
    runner = Runner(args)
    runner.start()
