"""Action callbacks for the frozen-base M2-4 residual agent."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from ..m2_3.features import state_to_features
from ..m2_3.model import TemporalDuelingDQN
from ..m2_3.planner import KnownHazardPlanner
from .config import (
    ACTIONS,
    ACTION_TO_INDEX,
    ALGORITHM_ID,
    BOMB_POWER,
    BRANCH,
    CHECKPOINT_FORMAT_VERSION,
    EPSILON_DECAY_STEPS,
    EPSILON_END,
    EPSILON_START,
    EVALUATION_MODEL_ENV,
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    M2_3_MODEL_PATH,
    MODEL_PATH,
    TEACHER_CONFIDENCE_THRESHOLD,
    TEACHER_ENABLED,
    TEACHER_ENEMY_DISTANCE,
    TEACHER_MIN_VOTES,
    TRAINING_SEED,
    VARIANT_ID,
    config_hash,
)
from .model import ResidualQNetwork, mask_q_values
from .teacher import RuleTeacher


def _load_torch_file(path, map_location, weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_source_metadata(metadata):
    required = {
        "variant_id": "M2-3",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                "Formal M2-3 source mismatch for {}: expected {!r}, found {!r}".format(
                    key, expected, metadata.get(key)
                )
            )


def _check_residual_metadata(metadata, source_sha256):
    required = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "variant_id": VARIANT_ID,
        "algorithm_id": ALGORITHM_ID,
        "branch": BRANCH,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
        "config_hash": config_hash(),
        "source_model_sha256": source_sha256,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                "Saved M2-4 model mismatch for {}: expected {!r}, found {!r}".format(
                    key, expected, metadata.get(key)
                )
            )
    if BRANCH == "c3" and not metadata.get("c1_source_sha256"):
        raise ValueError("Saved C3 model does not identify its selected C1 source")


def _evaluation_model_path():
    override = os.environ.get(EVALUATION_MODEL_ENV)
    if not override:
        return MODEL_PATH
    return Path(override).expanduser().resolve()


def _fresh_shield_window():
    return {
        "random_decisions": 0,
        "legal_actions": 0,
        "pruned_actions": 0,
        "bomb_candidates": 0,
        "bomb_pruned": 0,
        "no_safe_fallbacks": 0,
    }


def _fresh_teacher_window():
    return {
        "queries": 0,
        "accepted": 0,
        "low_confidence": 0,
        "illegal": 0,
        "agreement": 0,
        "irrelevant": 0,
        "unsafe": 0,
        "bomb_labels": 0,
        "useful_bomb_labels": 0,
        "confidence_sum": 0.0,
    }


def setup(self):
    """Load the frozen source model and prepare an independent residual head."""
    self.device = torch.device("cpu")
    self.rng = np.random.default_rng(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    self.temporal_planner = KnownHazardPlanner()
    self.shield_window = _fresh_shield_window()
    self.teacher_window = _fresh_teacher_window()
    self.teacher_records = {}
    self.rule_teacher = RuleTeacher() if self.train and TEACHER_ENABLED else None

    if not M2_3_MODEL_PATH.is_file():
        raise FileNotFoundError("M2-4 needs the formal M2-3 model at {}".format(M2_3_MODEL_PATH))
    source = _load_torch_file(M2_3_MODEL_PATH, self.device)
    _check_source_metadata(source.get("metadata", {}))
    source_state = source.get("model_state_dict")
    if source_state is None:
        raise ValueError("The formal M2-3 checkpoint has no model_state_dict")
    self.source_model_sha256 = _file_sha256(M2_3_MODEL_PATH)

    self.base_net = TemporalDuelingDQN(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.base_net.load_state_dict(source_state, strict=True)
    self.base_net.eval()
    for parameter in self.base_net.parameters():
        parameter.requires_grad_(False)

    self.residual_net = ResidualQNetwork(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.residual_net.eval()
    self.decision_step = 0

    with torch.no_grad():
        probe = torch.linspace(-1.0, 1.0, steps=4 * FEATURE_DIM).reshape(4, FEATURE_DIM)
        self.transfer_max_q_error = float(self.residual_net(probe).abs().max().item())

    if self.train:
        if BRANCH == "c3":
            self.logger.info("M2-4 C3 will continue from the selected C1 residual")
        else:
            self.logger.info("M2-4 %s starts from the frozen formal M2-3 model", BRANCH)
        return

    model_path = _evaluation_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            "No trained M2-4 {} model found at {}".format(BRANCH, model_path)
        )
    checkpoint = _load_torch_file(model_path, self.device)
    _check_residual_metadata(checkpoint.get("metadata", {}), self.source_model_sha256)
    residual_state = checkpoint.get("residual_state_dict")
    if residual_state is None:
        raise ValueError("The saved M2-4 model has no residual_state_dict")
    self.residual_net.load_state_dict(residual_state, strict=True)
    self.residual_net.eval()
    self.logger.info("Loaded M2-4 %s weights from %s", BRANCH, model_path)


def epsilon_for_step(decision_step):
    progress = min(max(decision_step / float(EPSILON_DECAY_STEPS), 0.0), 1.0)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def _network_values(self, features):
    states = torch.from_numpy(features).to(self.device).unsqueeze(0)
    self.base_net.eval()
    self.residual_net.eval()
    with torch.inference_mode():
        base_values = self.base_net(states)
        residual_values = self.residual_net(states)
    return base_values, base_values + residual_values


def _greedy_index(q_values, legal_mask):
    mask = torch.from_numpy(legal_mask).to(q_values.device).unsqueeze(0)
    return int(mask_q_values(q_values, mask).argmax(dim=1).item())


def _enemy_in_blast_line(game_state, origin):
    field = game_state["field"]
    opponents = {tuple(other[3]) for other in game_state["others"]}
    x, y = origin
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, BOMB_POWER + 1):
            tile = (x + dx * distance, y + dy * distance)
            if field[tile] == -1:
                break
            if tile in opponents:
                return True
            if field[tile] == 1:
                break
    return False


def _bomb_has_direct_use(game_state):
    field = game_state["field"]
    opponents = {tuple(other[3]) for other in game_state["others"]}
    x, y = tuple(game_state["self"][3])
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, BOMB_POWER + 1):
            tile = (x + dx * distance, y + dy * distance)
            if field[tile] == -1:
                break
            if tile in opponents or field[tile] == 1:
                return True
            if field[tile] != 0:
                break
    return False


def _teacher_relevance(game_state, base_index, teacher_index, analysis):
    if ACTIONS[base_index] == "BOMB" or ACTIONS[teacher_index] == "BOMB":
        return True
    if not bool(analysis.escape_feasible[base_index]):
        return True
    own_position = tuple(game_state["self"][3])
    opponents = [tuple(other[3]) for other in game_state["others"]]
    if opponents:
        nearest = min(
            abs(position[0] - own_position[0]) + abs(position[1] - own_position[1])
            for position in opponents
        )
        if nearest <= TEACHER_ENEMY_DISTANCE:
            return True
    return _enemy_in_blast_line(game_state, own_position)


def _teacher_record(self, game_state, base_index, analysis):
    vote = self.rule_teacher.query(game_state)
    window = self.teacher_window
    window["queries"] += 1
    window["confidence_sum"] += vote.confidence
    teacher_index = ACTION_TO_INDEX.get(vote.action)
    accepted = True
    reason = "accepted"

    if teacher_index is None:
        accepted, reason = False, "low_confidence"
    elif vote.confidence < TEACHER_CONFIDENCE_THRESHOLD or max(vote.votes) < TEACHER_MIN_VOTES:
        accepted, reason = False, "low_confidence"
    elif not bool(analysis.legal_mask[teacher_index]):
        accepted, reason = False, "illegal"
    elif teacher_index == base_index:
        accepted, reason = False, "agreement"
    elif not _teacher_relevance(game_state, base_index, teacher_index, analysis):
        accepted, reason = False, "irrelevant"
    elif not bool(analysis.escape_feasible[teacher_index]):
        accepted, reason = False, "unsafe"

    if accepted:
        window["accepted"] += 1
        if vote.action == "BOMB":
            window["bomb_labels"] += 1
            window["useful_bomb_labels"] += int(_bomb_has_direct_use(game_state))
    else:
        window[reason] += 1

    return {
        "accepted": accepted,
        "action": base_index if teacher_index is None else teacher_index,
        "confidence": float(vote.confidence),
        "votes": tuple(vote.votes),
        "reason": reason,
    }


def _state_key(game_state):
    return int(game_state["round"]), int(game_state["step"])


def act(self, game_state: dict) -> str:
    """Choose with total Q; the rule teacher never overrides this action."""
    if game_state is None:
        return "WAIT"

    analysis = self.temporal_planner.analyze(game_state, need_escape_labels=self.train)
    features = state_to_features(game_state, analysis)
    legal_mask = analysis.legal_mask
    legal_indices = np.flatnonzero(legal_mask)
    if legal_indices.size == 0:
        raise RuntimeError("The observed state has no legal action, including WAIT")

    base_values, total_values = _network_values(self, features)
    base_index = _greedy_index(base_values, legal_mask)
    if self.train and TEACHER_ENABLED:
        self.teacher_records[_state_key(game_state)] = _teacher_record(
            self, game_state, base_index, analysis
        )

    random_action = self.train and self.rng.random() < epsilon_for_step(self.decision_step)
    if random_action:
        safe_mask = legal_mask & analysis.escape_feasible
        safe_indices = np.flatnonzero(safe_mask)
        window = self.shield_window
        window["random_decisions"] += 1
        window["legal_actions"] += int(legal_mask.sum())
        window["pruned_actions"] += int((legal_mask & ~safe_mask).sum())
        if legal_mask[5]:
            window["bomb_candidates"] += 1
            window["bomb_pruned"] += int(not safe_mask[5])
        if safe_indices.size:
            action_index = int(self.rng.choice(safe_indices))
        else:
            window["no_safe_fallbacks"] += 1
            action_index = _greedy_index(total_values, legal_mask)
    else:
        action_index = _greedy_index(total_values, legal_mask)

    action = ACTIONS[action_index]
    if self.train and TEACHER_ENABLED:
        self.rule_teacher.commit_learner_action(game_state, action)
    if self.train:
        self.decision_step += 1
    return action
