import numpy as np
from tqdm import tqdm


def transform_move(move_idx, transform):
    from_sq = move_idx // 64
    to_sq = move_idx % 64

    def transform_sq(sq, t):
        row, col = divmod(sq, 8)
        if t == 'rot90':
            row, col = col, 7 - row
        elif t == 'rot180':
            row, col = 7 - row, 7 - col
        elif t == 'rot270':
            row, col = 7 - col, row
        elif t == 'fliplr':
            col = 7 - col
        elif t == 'flipud':
            row = 7 - row
        return row * 8 + col

    new_from = transform_sq(from_sq, transform)
    new_to = transform_sq(to_sq, transform)
    return new_from * 64 + new_to


def augment_state_move(state, move_idx):
    transforms = [
        ('id', lambda x: x, lambda m: m),
        ('rot90', lambda x: np.rot90(x, k=1, axes=(1, 2)),
         lambda m: transform_move(m, 'rot90')),
        ('rot180', lambda x: np.rot90(x, k=2, axes=(1, 2)),
         lambda m: transform_move(m, 'rot180')),
        ('rot270', lambda x: np.rot90(x, k=3, axes=(1, 2)),
         lambda m: transform_move(m, 'rot270')),
        ('fliplr', lambda x: np.flip(x, axis=2),
         lambda m: transform_move(m, 'fliplr')),
        ('flipud', lambda x: np.flip(x, axis=1),
         lambda m: transform_move(m, 'flipud')),
        ('rot90_fliplr', lambda x: np.flip(np.rot90(x, k=1, axes=(1, 2)), axis=2),
         lambda m: transform_move(transform_move(m, 'rot90'), 'fliplr')),
        ('rot90_flipud', lambda x: np.flip(np.rot90(x, k=1, axes=(1, 2)), axis=1),
         lambda m: transform_move(transform_move(m, 'rot90'), 'flipud'))
    ]
    result = []
    for _, trans_state, trans_move in transforms:
        new_state = trans_state(state.copy())
        new_move = trans_move(move_idx)
        result.append((new_state, new_move))
    return result


def main():
    in_path = "/data/dual_dataset_v1.npz"
    out_path = "D:/chess_project/ChessAnalyzer/src/data/chess_data/dual_dataset_augmented.npz"
    print("Loading original dataset...")
    data = np.load(in_path)
    states = data['states'].astype(np.float32)
    moves = data['moves']
    values = data['values']
    print(f"Original size: {len(states)}")

    all_states, all_moves, all_values = [], [], []
    for i in tqdm(range(len(states)), desc="Augmenting"):
        aug_list = augment_state_move(states[i], moves[i])
        for new_state, new_move in aug_list:
            all_states.append(new_state)
            all_moves.append(new_move)
            all_values.append(values[i])

    final_states = np.stack(all_states, axis=0)
    final_moves = np.array(all_moves)
    final_values = np.array(all_values)
    print(f"Augmented size: {len(final_states)}")

    np.savez_compressed(out_path,
                        states=final_states,
                        values=final_values,
                        moves=final_moves)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()