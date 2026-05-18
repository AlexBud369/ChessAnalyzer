import math
import numpy as np
import chess
from src.engine.policy_utils import legal_policy_probs, index_to_move

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def select_child(self, c_puct=1.4):
        best_score = -float('inf')
        best_child = None
        for move, child in self.children.items():
            score = child.value() + c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


class MCTS:
    def __init__(self, evaluator, num_iterations=400, c_puct=1.4):
        self.evaluator = evaluator
        self.num_iterations = num_iterations
        self.c_puct = c_puct
        self.reset_tree()

    def reset_tree(self):
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Vs = {}
        self.Es = {}

    def _hash_board(self, board: chess.Board) -> str:
        fen_parts = board.fen().split(' ')
        base_fen = ' '.join(fen_parts[:4])
        return base_fen

    def _action_to_idx(self, move: chess.Move) -> int:
        return move.from_square * 64 + move.to_square

    def search(self, board: chess.Board):
        s = self._hash_board(board)

        if board.is_game_over():
            if board.result() == "1-0":
                v = 1.0
            elif board.result() == "0-1":
                v = -1.0
            else:
                v = 0.0
            self.Es[s] = v
            return v

        if s not in self.Ps:
            result = self.evaluator.evaluate(board)
            value, policy_probs = result
            legal_probs = legal_policy_probs(policy_probs, board)
            self.Ps[s] = legal_probs
            self.Vs[s] = value
            self.Ns[s] = 0
            return value

        best_ucb = -float('inf')
        best_action = None
        for action in board.legal_moves:
            a = self._action_to_idx(action)
            q = self.Qsa.get((s, a), 0.0)
            u = self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa.get((s, a), 0))
            ucb = q + u
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        next_board = board.copy()
        next_board.push(best_action)
        v = self.search(next_board)

        a = self._action_to_idx(best_action)
        self.Nsa[(s, a)] = self.Nsa.get((s, a), 0) + 1
        self.Ns[s] += 1
        old_q = self.Qsa.get((s, a), 0.0)
        self.Qsa[(s, a)] = old_q + (v - old_q) / self.Nsa[(s, a)]

        return v

    def get_best_move(self, board: chess.Board):
        self.reset_tree()
        for _ in range(self.num_iterations):
            self.search(board)
        s = self._hash_board(board)
        best_action = None
        best_visits = -1
        for action in board.legal_moves:
            a = self._action_to_idx(action)
            visits = self.Nsa.get((s, a), 0)
            if visits > best_visits:
                best_visits = visits
                best_action = action
        return best_action

    def get_policy_prob(self, board: chess.Board, temperature=1.0):
        self.reset_tree()
        for _ in range(self.num_iterations):
            self.search(board)
        s = self._hash_board(board)
        policy = {}
        moves = list(board.legal_moves)
        if temperature == 0:
            best_move = max(moves, key=lambda m: self.Nsa.get((s, self._action_to_idx(m)), 0))
            for move in moves:
                policy[move] = 1.0 if move == best_move else 0.0
        else:
            visits = np.array([self.Nsa.get((s, self._action_to_idx(m)), 0) for m in moves], dtype=np.float32)
            if np.sum(visits) == 0:
                probs = np.ones(len(moves)) / len(moves)
            else:
                probs = visits ** (1.0 / temperature) / np.sum(visits ** (1.0 / temperature))
            for move, prob in zip(moves, probs):
                policy[move] = prob
        return policy

    def sample_move(self, board: chess.Board, temperature=1.0):
        policy = self.get_policy_prob(board, temperature=temperature)
        moves = list(policy.keys())
        probs = np.array([policy[m] for m in moves])

        probs = probs / np.sum(probs)
        chosen_idx = np.random.choice(len(moves), p=probs)
        return moves[chosen_idx]