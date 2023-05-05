import chess
import numpy as np

def getState(board):
    outcome = board.outcome()

    if outcome is None:
        return None
    
    winner = outcome.winner
    termination = outcome.termination

    if termination == chess.Termination.CHECKMATE:
        if winner:
            return 1
        return -1
    
    return 0

def heuristic_chess(board):
    if(getState(board) is not None):
        return getState(board)
    
    f_pawn =   len(board.pieces(chess.PAWN, chess.WHITE)) - len(board.pieces(chess.PAWN, chess.BLACK))
    f_knight =   len(board.pieces(chess.KNIGHT, chess.WHITE)) - len(board.pieces(chess.KNIGHT, chess.BLACK))
    f_bishop =   len(board.pieces(chess.BISHOP, chess.WHITE)) - len(board.pieces(chess.BISHOP, chess.BLACK))
    f_rook =   len(board.pieces(chess.ROOK, chess.WHITE)) - len(board.pieces(chess.ROOK, chess.BLACK))
    f_queen =   len(board.pieces(chess.QUEEN, chess.WHITE)) - len(board.pieces(chess.QUEEN, chess.BLACK))

    heuristic_value = (f_pawn + 3*f_knight + 4*f_bishop + 5*f_rook + 9*f_queen) / 100
    return heuristic_value

def is_cutoff(board, current_depth, depth_limit=2):
    if(getState(board) is not None or current_depth == depth_limit):
        return True
    return False

def h_minimax(board, depth_limit=2):
    return max_node(board, 0, depth_limit)

def max_node(board, current_depth, depth_limit):
    if(is_cutoff(board, current_depth, depth_limit=2)):
        return heuristic_chess(board), None
    
    v, move = -np.infty, -np.infty

    for a in board.legal_moves:
        board.push(a)
        v2, a2 = min_node(board, current_depth+1, depth_limit)
        board.pop()

        if v2 > v:
            v, move = v2, a
    
    return v, move

def min_node(board, current_depth, depth_limit):
    if(is_cutoff(board, current_depth, depth_limit=2)):
        return heuristic_chess(board), None
    
    v, move = np.infty, np.infty

    for a in board.legal_moves:
        board.push(a)
        v2, a2 = max_node(board, current_depth+1, depth_limit)
        board.pop()

        if v2 < v:
            v, move = v2, a
    
    return v, move

def h_minimax_alpha_beta(board, depth_limit=2):
    return max_node_ab(board, 0, depth_limit, -np.infty, np.infty)

def max_node_ab(board, current_depth, depth_limit, alpha, beta):
    if(is_cutoff(board, current_depth, depth_limit=2)):
        return heuristic_chess(board), None
    
    v = -np.infty

    for a in board.legal_moves:
        board.push(a)
        v2, a2 = min_node_ab(board, current_depth+1, depth_limit, alpha, beta)
        board.pop()

        if v2 > v:
            v, move = v2, a
            alpha = max(alpha, v)
        
        if v >= beta:
            return v, move
    
    return v, move

def min_node_ab(board, current_depth, depth_limit, alpha, beta):
    if(is_cutoff(board, current_depth, depth_limit=2)):
        return heuristic_chess(board), None
    
    v = np.infty

    for a in board.legal_moves:
        board.push(a)
        v2, a2 = max_node_ab(board, current_depth+1, depth_limit, alpha, beta)
        board.pop()

        if v2 < v:
            v, move = v2, a
            beta = min(beta, v)
        
        if v <= alpha:
            return v, move
    
    return v, move