import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from packed_dataset import pack_dataset_init

INPUT_DIR = "dataset_init"
OUTPUT_PATH = "dataset_init.npz"


def main():
    pack_dataset_init(INPUT_DIR, OUTPUT_PATH)


if __name__ == "__main__":
    main()
