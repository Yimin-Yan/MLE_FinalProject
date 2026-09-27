import os
import uuid
import datetime
import gzip
import pickle
from typing import Any, Dict, List



# -------------------------------------------------------------------
# In your callbacks.py
# -------------------------------------------------------------------
def setup_training(self):
    """
    Initialise self for recording transitions.
    """

    # generate unique run name and folder
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    self.run_name = f"run_{now}_{short_id}"
    self.run_folder = os.path.join("runs", self.run_name)
    os.makedirs(self.run_folder, exist_ok=True)


    # path for *all* transitions
    self.transitions_file = os.path.join(self.run_folder, "transitions_all_games.pkl.gz")
    # ensure it's fresh
    with gzip.open(self.transitions_file, "wb") as f:
        pass


    # uffer to collect transitions for this round
    self._transitions: List[Dict] = []
    self.game = 1

def game_events_occurred(self,
                         old_game_state: dict,
                         self_action: str,
                         new_game_state: dict,
                         events: List[str]):
    """
    Record the transition
    """
    # RECORDING LOGIC:

    # RECORD FULL OLD STATE + action + events
    self._transitions.append({
        "state": old_game_state,
        "action": self_action,
        "events": events
    })

def end_of_round(self,
                 last_game_state: dict,
                 last_action: str,
                 events: List[str]):
    """
    At round end: record the final transition, then dump all to a compressed file.
    """
    # record the very final transition
    self._transitions.append({
        "state": last_game_state,
        "action": last_action,
        "events": events
    })

    # -------- save everything into this run's folder --------
    # transitions
    # append this game’s transitions to the one file
    with gzip.open(self.transitions_file, "ab") as f:
        # include game index for easy splitting later
        pickle.dump({"game": self.game, "transitions": self._transitions}, f)

    # clear buffer for next round
    self._transitions.clear()
    self._prev_state = None
    self.game += 1
