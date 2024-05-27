# Score Follower 

## Description
This repository contains the Score Follower application developed as part of a master's project. This product serves as a first application which uses Gaussian Processes to perform score following. 

## Installation
- The score follower is designed for evaluation using the adapted score renderer app available here: https://github.com/josephinecowley/Score-follower.

### Prerequisites
- A pre-recorded .wav file if running in Pre-recorded Mode.
- Score in MIDI format.
- Score in MusicXML format.
- Dependencies listed in `requirements.txt`

### Steps
1. Download the adapted Flippy Qualitative Testbench in the Prerequisites section.
2. Update the parameters and GP hyperparameters in the file `args.py`.
4. Follow instructions in https://github.com/flippy-fyp/flippy-qualitative-testbench/blob/main/README.md to use the score renderer as an evaluation tool of this score follower.
5. ```bash
   python scorefollow.py

Adjust the command with the appropriate flags or input parameters as needed for your specific setup.

## Contributing
Contributions to the Score Follower project are welcome. Please follow the steps below to contribute:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/your_feature_name`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your_feature_name`).
5. Create a new Pull Request.



