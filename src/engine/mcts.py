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
    def __init__(self, evaluator, num_iterations=800):
        self.evaluator = evaluator
        self.num_iterations = num_iterations

    def search(self, root_board: chess.Board):
        root = MCTSNode(root_board)
        for _ in range(self.num_iterations):
            node = root

            while node.is_expanded():
                node = node.select_child()

            value, policy_probs = self.evaluator.evaluate(node.board)
            legal_probs = legal_policy_probs(policy_probs, node.board)

            for move in node.board.legal_moves:
                idx = move.from_square * 64 + move.to_square
                prior = legal_probs[idx]
                if prior > 1e-6:
                    new_board = node.board.copy()
                    new_board.push(move)
                    node.children[move] = MCTSNode(new_board, parent=node, prior=prior)

            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

        best_move = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        return best_move