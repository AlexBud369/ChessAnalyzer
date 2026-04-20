import chess
from typing import Optional, Tuple

class TranspositionTable:
    def __init__(self, size_mb: int = 128):
        self.size = size_mb * 1024 * 1024 // 32
        self.table = [None] * self.size
        self.new_entries = 0

    def _get_hash(self, board: chess.Board) -> int:
        if hasattr(board, 'zobrist_hash'):
            return board.zobrist_hash()
        if hasattr(board, 'transposition_key'):
            key = board.transposition_key()

            if isinstance(key, tuple):
                return key[0] ^ key[1]
            return key
        if hasattr(board, '_transposition_key'):
            key = board._transposition_key()
            if isinstance(key, tuple):
                return key[0] ^ key[1]
            return key
        return hash(board.fen())

    def _hash(self, board: chess.Board) -> int:
        return self._get_hash(board) % self.size

    def _full_hash(self, board: chess.Board) -> int:
        return self._get_hash(board)

    def store(self, board: chess.Board, depth: int, eval: float, move: Optional[chess.Move],
              node_type: str, alpha: float, beta: float):
        key = self._hash(board)
        entry = {
            'depth': depth,
            'eval': eval,
            'move': move,
            'node_type': node_type,
            'hash': self._full_hash(board),
        }
        old = self.table[key]
        if old is None or old['depth'] <= depth:
            self.table[key] = entry
            self.new_entries += 1

    def lookup(self, board: chess.Board, depth: int, alpha: float, beta: float
               ) -> Tuple[Optional[float], Optional[chess.Move]]:
        key = self._hash(board)
        entry = self.table[key]

        if entry is None or entry['hash'] != self._full_hash(board):
            return None, None
        if entry['depth'] < depth:
            return None, entry.get('move')
        if entry['node_type'] == 'exact':
            return entry['eval'], entry.get('move')
        elif entry['node_type'] == 'lower' and entry['eval'] >= beta:
            return entry['eval'], entry.get('move')
        elif entry['node_type'] == 'upper' and entry['eval'] <= alpha:
            return entry['eval'], entry.get('move')
        return None, entry.get('move')

    def clear(self):
        self.table = [None] * self.size
        self.new_entries = 0