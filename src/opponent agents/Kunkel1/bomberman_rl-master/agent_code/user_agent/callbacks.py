from q_helpers import state_to_tabular_features, describe_tabular_state, state_to_cnn_features, describe_cnn_state
import os
from datetime import datetime
import pickle
import numpy as np
import pygame

def create_run_folder(base="snapshots"):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def setup(self):
    self.run_dir = create_run_folder()
    self.step = 0



def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    features = state_to_cnn_features(game_state)


    # Save game_state dict
    with open(os.path.join(self.run_dir, f"state_{self.step:05d}.pkl"), "wb") as f:
        pickle.dump(game_state, f)

    # Save features as numpy array
    np.save(os.path.join(self.run_dir, f"features_{self.step:05d}.npy"), features)

    screen = pygame.display.get_surface()
    # Save screenshot of current screen
    pygame.image.save(screen, os.path.join(self.run_dir, f"screenshot_{self.step:05d}.png"))


    self.step += 1
    return game_state['user_input']
