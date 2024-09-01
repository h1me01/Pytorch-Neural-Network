import torch
import torch.nn as nn
import chess
import chess.pgn
import sparse
import math

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
weights_path = 'weights/nn-e6b128-768-512-1.nnue'
net = ChessNN()
net.load_state_dict(torch.load(weights_path, weights_only=True))
net.eval()

def evaluate_position(board):
    fen = board.fen()
    input_fen = sparse.get(fen)
    input_fen = torch.tensor(input_fen, dtype=torch.float32).unsqueeze(0)
    net.eval()
    
    with torch.no_grad():
        output = net(input_fen)
    output = output.squeeze() 
       
    return output.item()

def quiescence_search(board, alpha, beta, node_count):
    stand_pat = evaluate_position(board)
    node_count[0] += 1
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiescence_search(board, -beta, -alpha, node_count)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha

def negamax(board, depth, alpha, beta, node_count):
    if depth == 0 or board.is_game_over():
        return quiescence_search(board, alpha, beta, node_count)
    
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, node_count)
        board.pop()
        node_count[0] += 1
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return alpha

def iterative_deepening(board, max_depth):
    best_move = None
    best_score = -math.inf

    for depth in range(1, max_depth + 1):
        node_count = [0]
        current_best_move = None
        current_best_score = -math.inf
        alpha = -math.inf
        beta = math.inf
        for move in board.legal_moves:
            board.push(move)
            score = -negamax(board, depth - 1, -beta, -alpha, node_count)
            board.pop()
            if score > current_best_score:
                current_best_score = score
                current_best_move = move
            alpha = max(alpha, score)
        best_move = current_best_move
        best_score = current_best_score
        
        print(f'Depth {depth}: Best move {best_move}, Score {best_score/100.0:.2f}, Nodes: {node_count[0]}')
    return best_move, best_score

def play_game(starting_fen, max_depth):
    board = chess.Board()
    board.set_fen(starting_fen)

    move_accumulator = ""
    while not board.is_game_over():
        print("Ply: ", board.ply())
        best_move, score = iterative_deepening(board, max_depth)
        piece = board.piece_at(best_move.from_square)
        move_accumulator += piece.symbol().upper() + str(best_move) + " "

        board.push(best_move)     
        print(f'\nMove played: {best_move}, Score: {score/100.0:.2f}\n')
        print(board)
        print("")
        print(move_accumulator)
        print("")

        if board.is_repetition(3):
            print("Draw by repetition!")
            break

if __name__ == '__main__':
    fen = '3KR3/8/2k5/8/8/8/8/8 w - - 0 1'
    board = chess.Board()
    board.set_fen(fen)
    
    max_depth = 10
    best_move, score = iterative_deepening(board, max_depth)

    #play_game(fen, max_depth)
