import sys
import gzip
import pickle
from tqdm import tqdm
import os
import tensorflow as tf

# adjust if your project structure differs:
from q_helpers import state_to_tabular_features, state_to_cnn_features, reward_from_events, ACTS, ACT_BITS, TransitionFields, get_legal_actions


def build_and_write_shards(folder: str, out_dir: str, shard_size: int = 500, use_cnn: bool = False, compress: bool = True):
    """
    Reads transitions_all_games.pkl.gz and emits shards of `shard_size` episodes.
    Each shard is a *separate pickle* containing a list[episode], so it can be loaded independently.
    """
    os.makedirs(out_dir, exist_ok=True)
    infile = os.path.join(folder, "transitions_all_games.pkl.gz")

    shard, shard_idx, written = [], 0, 0

    open_out = (lambda p: gzip.open(p, "wb")) if compress else (lambda p: open(p, "wb"))

    with gzip.open(infile, "rb") as fin:
        pbar = tqdm(desc="Building episodes", unit="game")
        while True:
            try:
                rec = pickle.load(fin)
            except EOFError:
                break

            transitions = rec["transitions"]
            episode = []
            for idx, step in enumerate(transitions):
                state         = state_to_cnn_features(step["state"]) if use_cnn else state_to_tabular_features(step["state"])
                legal_actions = get_legal_actions(step["state"])
                action        = ACT_BITS[step["action"]]
                reward        = reward_from_events(step["events"])
                done          = (idx == len(transitions) - 1)
                episode.append([idx, tf.convert_to_tensor(state[tf.newaxis, :], dtype=tf.float32), legal_actions, action, reward, done])

            # fill next_state / next_legal_actions
            for i in range(len(episode) - 1):
                next_state         = episode[i + 1][TransitionFields.STATE]
                next_legal_actions = episode[i + 1][TransitionFields.LEGAL_ACTIONS]
                episode[i].extend([next_state, next_legal_actions])

            # terminal next_state
            episode[-1].extend([None, None])

            shard.append(episode)
            written += 1

            if len(shard) >= shard_size:
                shard_idx += 1
                out_name = f"episodes-shard-{shard_idx:05d}.pkl" + (".gz" if compress else "")
                out_path = os.path.join(out_dir, out_name)
                with open_out(out_path) as fout:
                    pickle.dump(shard, fout)
                shard.clear()

            pbar.update(1)

        pbar.close()

    # flush last partial shard
    if shard:
        shard_idx += 1
        out_name = f"episodes-shard-{shard_idx:05d}.pkl" + (".gz" if compress else "")
        out_path = os.path.join(out_dir, out_name)
        with open_out(out_path) as fout:
            pickle.dump(shard, fout)

    print(f"Wrote {shard_idx} shard(s), {written} episodes, to {out_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_training_episodes.py <path/to/transitions> [--cnn]")
        sys.exit(1)

    out_path = sys.argv[1]
    use_cnn = ("--cnn" in sys.argv)
    mode = "CNN" if use_cnn else "tabular"
    print(f"Building {mode} transitions...")
    n = build_and_write_shards(out_path, out_path, use_cnn=use_cnn)
    print(f"Built and saved {n} {mode} episodes → {out_path}")

if __name__ == "__main__":
    main()
