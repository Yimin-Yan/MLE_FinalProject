import os
import sys
import gzip
import pickle
from tqdm import tqdm
from q_helpers import reward_from_events, TransitionFields

def update_rewards_in_transitions(folder: str):
    """
    Load both transitions.pkl and transitions_all_games.pkl.gz,
    updates rewards using reward_from_events, and saves the updated transitions.pkl.
    """


    transitions_pkl_path = os.path.join(folder, "transitions.pkl")
    if not os.path.isfile(transitions_pkl_path):
        print(f"File not found: {transitions_pkl_path}")
        sys.exit(1)

    # Load the transitions file (the one with just transitions, not full states)
    with open(transitions_pkl_path, 'rb') as f:
        episodes = pickle.load(f)

    # Load the original transitions_all_games.pkl.gz to get state and events info
    transitions_all_path = os.path.join(folder, "transitions_all_games.pkl.gz")

    if not os.path.isfile(transitions_all_path):
        print(f"File not found: {transitions_all_path}")
        sys.exit(1)

    with gzip.open(transitions_all_path, 'rb') as f:
        pbar = tqdm(desc="Loading original transitions", unit="game")
        original_data = []
        while True:
            try:
                rec = pickle.load(f)
                original_data.append(rec)
                pbar.update(1)
            except EOFError:
                break
        pbar.close()

    # Now, we need to update rewards in the loaded episodes
    updated_episodes = []
    print("Updating rewards...")

    # Iterate through the episodes and update rewards
    for episode_idx, episode in tqdm(enumerate(episodes), desc="Episodes", unit="episode"):
        # Get the corresponding transitions from the original data
        transitions = original_data[episode_idx]['transitions']

        for idx, transition in enumerate(episode):
            # Make sure we have the matching transition from original data
            original_transition = transitions[idx]
            events = original_transition['events']

            # Update the reward using the reward_from_events function
            new_reward = reward_from_events(events)
            transition[TransitionFields.REWARD] = new_reward

        updated_episodes.append(episode)

    # Save the updated episodes back to transitions.pkl
    with open(transitions_pkl_path, 'wb') as f:
        pickle.dump(updated_episodes, f)

    print(f"Updated rewards for {len(updated_episodes)} episodes → {transitions_pkl_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python update_rewards.py <path/to/transitions_folder>")
        sys.exit(1)

    folder = sys.argv[1]

    update_rewards_in_transitions(folder)


if __name__ == "__main__":
    main()
