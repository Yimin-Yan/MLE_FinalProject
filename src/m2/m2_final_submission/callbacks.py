"""Self-contained stage-conditioned V20 inference policy."""

from collections import deque
import random
from types import SimpleNamespace

import numpy as np
import torch

from .config import ACTIONS, MOVE_DELTAS, RANDOM_SEED, configured_device, configured_model_path, configured_stage
from .features import state_to_features, valid_action_mask
from .model import DuelingQNetwork, load_checkpoint, load_policy_state
from .planner import BOMB_INDEX, exact_action_mask, exact_survival_metrics, tactical_action
from .rule_policy import act as rule_act, setup as rule_setup


def setup(self):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    self.rng = np.random.default_rng(RANDOM_SEED)
    requested = configured_device()
    cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if requested == "cuda" or (requested == "auto" and cuda) else "cpu")
    if requested == "cuda" and not cuda:
        raise RuntimeError("M2_DEVICE=cuda but CUDA is unavailable")
    self.stage = configured_stage()
    self.model_path = configured_model_path()
    self.policy_net = DuelingQNetwork(len(ACTIONS)).to(self.device)
    self.stage_steps = 0
    self.optimizer_steps = 0
    self.position_history = deque(maxlen=20)
    self.history_round = None
    if self.model_path.is_file():
        payload = load_checkpoint(self.model_path, self.device)
        self.checkpoint_migrated = load_policy_state(self.policy_net, payload["policy_state"])
        self.stage_steps = int(payload.get("stage_steps", 0))
        self.optimizer_steps = int(payload.get("optimizer_steps", 0))
    elif not self.train:
        raise FileNotFoundError(f"M2 checkpoint not found: {self.model_path}")
    else:
        self.checkpoint_migrated = False
    self.policy_net.eval()
    self.task3a_rule_context = None
    if self.stage == "task3a":
        self.task3a_rule_context = SimpleNamespace(logger=self.logger)
        rule_setup(self.task3a_rule_context)


def act(self, game_state: dict):
    if self.stage == "task3a":
        return _act_task3a(self, game_state)
    if self.stage == "task1":
        return _act_task1(self, game_state)
    return _act_exact(self, game_state)


def _act_task3a(self, game_state):
    proposal = rule_act(self.task3a_rule_context, game_state)
    physical = valid_action_mask(game_state)
    if proposal in ACTIONS:
        index = ACTIONS.index(proposal)
        if physical[index]:
            if proposal == "BOMB":
                return proposal
            survivable, _ = exact_survival_metrics(game_state)
            if survivable[index]:
                return proposal
    return _act_exact(self, game_state)


def _act_task1(self, game_state):
    legal, _ = exact_action_mask(game_state)
    legal[BOMB_INDEX] = False
    legal = _apply_loop_mask(self, game_state, legal)
    if not np.any(legal):
        raise RuntimeError("No legal M2 Task1 retention action")
    q_values = _q_values(self, game_state)
    q_values[~legal] = -np.inf
    override = tactical_action(game_state, legal, q_values, self.rng)
    if override is not None:
        return ACTIONS[override]
    best = np.flatnonzero(legal & (q_values == np.max(q_values)))
    return ACTIONS[int(self.rng.choice(best))]


def _act_exact(self, game_state):
    legal, _ = exact_action_mask(game_state)
    legal = _apply_loop_mask(self, game_state, legal)
    if not np.any(legal):
        raise RuntimeError("No legal M2 tactical action")
    q_values = _q_values(self, game_state)
    q_values[~legal] = -np.inf
    override = tactical_action(game_state, legal, q_values, self.rng)
    if override is not None:
        return ACTIONS[override]
    best = np.flatnonzero(legal & (q_values == np.max(q_values)))
    return ACTIONS[int(self.rng.choice(best))]


def _q_values(self, game_state):
    board, aux = state_to_features(game_state)
    with torch.inference_mode():
        return self.policy_net(
            torch.from_numpy(board).unsqueeze(0).to(self.device),
            torch.from_numpy(aux).unsqueeze(0).to(self.device),
        )[0].detach().cpu().numpy()


def _apply_loop_mask(self, game_state, legal):
    round_id = int(game_state["round"])
    if self.history_round != round_id:
        self.position_history.clear()
        self.history_round = round_id
    position = tuple(game_state["self"][3])
    repeated = self.position_history.count(position) > 2
    self.position_history.append(position)
    if not repeated:
        return legal
    recent = list(self.position_history)[-10:]
    less_visited = legal.copy()
    for index, action in enumerate(ACTIONS[:5]):
        if not less_visited[index]:
            continue
        if action == "WAIT":
            destination = position
        else:
            dx, dy = MOVE_DELTAS[action]
            destination = (position[0] + dx, position[1] + dy)
        if recent.count(destination) >= 2:
            less_visited[index] = False
    if np.any(less_visited[:5]):
        return less_visited
    return legal
