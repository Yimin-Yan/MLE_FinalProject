import sys
import gzip
import pickle
from tqdm import tqdm
import os

# adjust if your project structure differs:
from q_helpers import state_to_tabular_features, state_to_cnn_features, reward_from_events, ACTS, ACT_BITS, TransitionFields, get_legal_actions

PICKLE_PROTOCOL = 2  # protocol 2 keeps things simple (no frames)

def _strip_pickle_outer(pickled: bytes) -> bytes:
    """
    Convert a full protocol-2 pickle of a single object into its inner opcodes,
    removing the leading PROTO(2) and trailing STOP so it can be injected
    inside a MARK/APPENDS block.
    """
    if not pickled:
        return pickled
    if pickled[:2] != b'\x80\x02':  # PROTO 2 header
        raise ValueError("Streaming writer expects protocol=2 pickles.")
    if pickled[-1:] != b'.':        # STOP
        raise ValueError("Pickle must end with STOP opcode.")
    return pickled[2:-1]

def build_and_stream(folder: str, out_path: str, use_cnn: bool = False) -> int:
    """
    Read transitions from 'transitions_all_games.pkl.gz', build episodes one by one,
    and stream-append each episode into a single on-disk pickle list at out_path.
    If use_cnn=True, use CNN features instead of tabular features.
    Returns the number of episodes written.
    """
    infile = os.path.join(folder, "transitions_all_games.pkl.gz")

    # Opcodes we’ll emit (protocol 2):
    # PROTO 2:    b'\x80\x02'
    # EMPTY_LIST: b']'
    # MARK:       b'('
    # APPENDS:    b'e'
    # STOP:       b'.'

    written = 0
    with gzip.open(infile, 'rb') as fin, open(out_path, 'wb') as fout:
        # Start pickle stream with an empty list
        fout.write(b"\x80\x02")  # PROTO 2
        fout.write(b"]")         # EMPTY_LIST

        pbar = tqdm(desc="Processing games", unit="game")
        while True:
            try:
                rec = pickle.load(fin)
            except EOFError:
                break

            transitions = rec['transitions']

            # Build one episode entirely in memory (bounded)
            episode = []
            for idx, step in enumerate(transitions):
                # Use tabular or CNN features depending on flag
                if use_cnn:
                    state = state_to_cnn_features(step['state'])
                else:
                    state = state_to_tabular_features(step['state'])
                legal_actions = get_legal_actions(step['state'])
                action        = ACT_BITS[step['action']]
                if action is None:
                    raise ValueError("action == None")

                reward        = reward_from_events(step['events'])
                done          = (idx == len(transitions) - 1)
                transition    = [idx, state, legal_actions, action, reward, done]
                episode.append(transition)

            # Add next_state / next_legal_actions
            for i in range(len(episode) - 1):
                next_state         = episode[i + 1][TransitionFields.STATE]
                next_legal_actions = episode[i + 1][TransitionFields.LEGAL_ACTIONS]
                episode[i].extend([next_state, next_legal_actions])
            # Terminal next state for cnn should still be tensor-shaped for batching
            if use_cnn:
                next_state = state_to_cnn_features(None)
            else:
                next_state = None
            episode[-1].extend([next_state, None])  # terminal

            # Stream-append this episode to the on-disk list
            inner = _strip_pickle_outer(pickle.dumps(episode, protocol=PICKLE_PROTOCOL))
            fout.write(b"(")   # MARK
            fout.write(inner)  # push one item
            fout.write(b"e")   # APPENDS (extend list with all items since MARK) => 1 ep appended

            written += 1
            pbar.update(1)

        pbar.close()
        fout.write(b".")  # STOP

    return written

def main():
    if len(sys.argv) < 2:
        print("Usage: python build_training_episodes.py <path/to/transitions> [--cnn]")
        sys.exit(1)

    folder = sys.argv[1]
    use_cnn = ("--cnn" in sys.argv)
    mode = "CNN" if use_cnn else "tabular"
    print(f"Building {mode} transitions...")
    out_path = os.path.join(folder, "transitions_cnn.pkl" if use_cnn else "transitions.pkl")
    n = build_and_stream(folder, out_path, use_cnn=use_cnn)
    print(f"Built and saved {n} {mode} episodes → {out_path}")

if __name__ == "__main__":
    main()
