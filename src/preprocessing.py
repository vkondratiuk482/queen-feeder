import chess
import numpy as np

UNIQUE_PIECES = 6
SQUARES_COUNT = 64

# moves - string of moves separated by space, e.g: e4 d5 .....
def preprocess_game(moves_line):
    board = chess.Board()
    moves = moves_line.split(' ')

    X = []
    y = []

    for move in moves:
        X.append(__board_to_vector(board))
        y.append(__encode_predicted_move(board, move))

        board.push_san(move)

    return np.array(X), np.array(y)

def __board_to_vector(board): 
    vector = np.zeros(SQUARES_COUNT, dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            vector[square] = __encode_piece(piece_type=piece.piece_type, is_white=piece.color)

    return vector

def __encode_predicted_move(board, predicted_move):
    move = board.parse_san(predicted_move)

    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)

    return __encode_position(from_square + to_square)

def __encode_piece(piece_type, is_white):
    if is_white:
        return piece_type

    return piece_type + UNIQUE_PIECES

def __encode_position(move):
    from_position = __get_square_index(move[0:2])
    to_position = __get_square_index(move[2:4])

    return from_position * SQUARES_COUNT + to_position

def __decode_position(encoded):
    to_position = encoded % SQUARES_COUNT
    from_position = encoded // SQUARES_COUNT

    return (from_position, to_position)

def __get_square_index(square):
    letter_ascii = ord(square[0])
    digit = int(square[1])

    return (letter_ascii - ord('a')) + (8 * (digit - 1)) 