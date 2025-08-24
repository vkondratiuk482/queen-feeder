import chess
import numpy as np

from typing import Tuple, Dict, List
from numpy.typing import NDArray
from constants import MOVES_DELIMITER, UNIQUE_PIECES, BOARD_COLUMNS, BOARD_ROWS, FRIENDLY_PIECE, ENEMY_PIECE

def preprocess_game(moves_line: str) -> Tuple[
    NDArray,
    NDArray,
    Dict[int, List[Tuple[NDArray, NDArray]]],
]:
    board = chess.Board()
    moves = moves_line.split(MOVES_DELIMITER)

    piece_selection_X: List[NDArray] = []
    piece_selection_y: List[NDArray] = []

    move_selection: Dict[int, List[Tuple[NDArray, NDArray]]] = {}

    for index, move in enumerate(moves):
        # TODO: implement board-flipping logic
        # to train on both white & black moves
        if index % 2 != 0:
            board.push_san(move)
            continue

        input_matrices = board_to_matrices(board)
        piece_selection_X.append(input_matrices)
        from_square = get_move_from(board, move)
        piece_selection_y.append(from_square)

        piece_type = board.piece_at(from_square).piece_type - 1
        if not piece_type in move_selection:
            move_selection[piece_type] = []
        move_selection[piece_type].append((input_matrices, get_move_to(board, move)))
        
        board.push_san(move)

    return np.array(piece_selection_X), np.array(piece_selection_y), move_selection

def board_to_matrices(board: chess.Board) -> NDArray:
    X = np.zeros(shape=[UNIQUE_PIECES, BOARD_ROWS, BOARD_COLUMNS])

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if not piece:
            continue

        row = square // BOARD_ROWS
        column = square % BOARD_COLUMNS
        piece_type = piece.piece_type - 1

        X[piece_type, row, column] = FRIENDLY_PIECE if piece.color else ENEMY_PIECE

    return X

def get_move_from(board: chess.Board, actual_move: str) -> int:
    move = board.parse_san(actual_move)

    return move.from_square

def get_move_to(board: chess.Board, actual_move: str) -> int:
    move = board.parse_san(actual_move)

    return move.to_square