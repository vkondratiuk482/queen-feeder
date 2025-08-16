import re
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from preprocessing import preprocess_game

TINY_FILE_PATH = '../data/tiny.csv'
TINY_MOVES_COLUMN = 'moves'
LARGE_FILE_PATH = '../data/large.csv'
LARGE_MOVES_COLUMN = 'AN'

class PreprocessedDataset(Dataset):
    def __init__(self, only_tiny=False, tiny_limit = 20_000, large_limit = 1_000_000):
        (tiny_X, tiny_y) = load_tiny_dataset(tiny_limit)

        if only_tiny:
            self.X = tiny_X
            self.y = tiny_y

            return

        (large_X, large_y) = load_large_dataset(large_limit)

        self.X = np.concatenate((tiny_X, large_X), axis=0)
        self.y = np.concatenate((tiny_y, large_y), axis=0)

        return

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_tiny_dataset(limit):
    tiny_df = pd.read_csv(TINY_FILE_PATH, nrows=limit)

    tiny_X = []
    tiny_y = []

    for index, raw in enumerate(tiny_df[TINY_MOVES_COLUMN]):
        (X, y) = preprocess_game(raw)

        print(f"Loaded & preprocessed game from tiny dataset {index}/{len(tiny_df)}")

        tiny_X.append(X)
        tiny_y.append(y)

    return (np.concatenate(tiny_X, axis=0), np.concatenate(tiny_y, axis=0))

def load_large_dataset(limit):
    large_df = pd.read_csv(LARGE_FILE_PATH, nrows=limit)

    large_X = []
    large_y = []

    for index, raw in enumerate(large_df[LARGE_MOVES_COLUMN]):
        adapted = large_adapter(raw)

        try:
            (X, y) = preprocess_game(adapted)

            print(f"Loaded & preprocessed game from large dataset {index}/{len(large_df)}")

            large_X.append(X)
            large_y.append(y)
        except Exception as ex: # some moves are still not compatible
            continue 
    
    return (np.concatenate(large_X, axis=0), np.concatenate(large_y, axis=0))

def large_adapter(game_history):
    no_numbers = re.sub(r'\d+\.', '', game_history)
        
    moves = [move.strip() for move in no_numbers.split() if move.strip()]
    moves.pop() # remove winner / loser

    return ' '.join(moves)