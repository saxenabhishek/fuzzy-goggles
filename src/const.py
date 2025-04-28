import os
import pickle
from typing import Dict

WINDOW_LENGTH = 25
DIR = "./techLargCapStock"
TRAINING_ENVS = 6


def read_dict_from_pickle(filename: str, input_dir: str = ".") -> Dict:
    file_path = os.path.join(input_dir, f"{filename}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    try:
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)
        print(f"Read data from {file_path}")
        return data_dict
    except Exception as e:
        raise Exception(f"Error reading from pickle file: {e}")
