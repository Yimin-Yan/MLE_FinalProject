"""Training callbacks for the matched M2-1 continuation control."""

from dataclasses import dataclass
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import events as e

from ..m2_1.features import legal_action_mask, potential_from_features, state_to_features
from ..m2_1.model import DuelingDQN, mask_q_values
from ..m2_1.n_step import NStepAccumulator, OneStepTransition
from ..m2_1.replay_buffer import LAPReplayBuffer
from .callbacks import _load_torch_file, epsilon_for_step
from .config import (
    ACTIONS,
    ACTION_TO_INDEX,
    ADAM_BETAS,
    ADAM_EPS,
    BATCH_SIZE,
    CHECKPOINT_FORMAT_VERSION,
    FAIL_ON_STATIC_INVALID,
    FEATURE_DIM,
    FULL_CHECKPOINT_INTERVAL,
    GAMMA,
    GRAD_CLIP_NORM,
    LAP_ALPHA,
    LAP_PRIORITY_FLOOR,
    LEARNING_RATE,
    M2_1_MODEL_PATH,
    MAX_TRAINING_TRANSITIONS,
    METRIC_LOG_INTERVAL,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_SNAPSHOT_DIR,
    MODEL_SNAPSHOT_INTERVAL,
    N_STEP,
    PENALTY_KILLED,
    PENALTY_RACE_INVALID,
    PENALTY_STATIC_INVALID,
    PENALTY_SUICIDE,
    POTENTIAL_BETA,
    REPLAY_CAPACITY,
    REPLAY_PERIOD,
    RESUME_TRAINING,
    REWARD_COIN,
    REWARD_CRATE,
    REWARD_KILL,
    TARGET_UPDATE_INTERVAL,
    TRAINING_CHECKPOINT_PATH,
    USE_POTENTIAL_SHAPING,
    VARIANT_ID,
    WARMUP_TRANSITIONS,
    checkpoint_metadata,
    config_hash,
)


@dataclass
class PendingTransition:
    state: np.ndarray
    action_index: int
    action_name: str
    next_state: np.ndarray
    next_mask: np.ndarray
    old_mask: np.ndarray
    round_number: int
    step_number: int
    events: tuple


def _fresh_metric_window():
    return {
        "updates": 0,
        "loss_sum": 0.0,
        "td_abs_sum": 0.0,
        "td_abs_max": 0.0,
        "grad_norm_sum": 0.0,
        "q_sum": 0.0,
        "q_count": 0,
        "q_abs_max": 0.0,
        "sample_priority_sum": 0.0,
    }


def setup_training(self):
    """Create a fresh optimizer/replay pair and load the same M2-1 start."""
    self.target_net = DuelingDQN(
        input_dim=FEATURE_DIM,
        action_dim=len(ACTIONS),
    ).to(self.device)
    self.optimizer = torch.optim.Adam(
        self.online_net.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
    )
    self.replay_buffer = LAPReplayBuffer(
        REPLAY_CAPACITY,
        FEATURE_DIM,
        LAP_ALPHA,
        LAP_PRIORITY_FLOOR,
    )
    self.n_step_accumulator = NStepAccumulator(N_STEP, GAMMA)
    self.pending_transition = None
    self.source_transition_count = 0
    self.transition_count = 0
    self.optimizer_step = 0
    self.rounds_completed = 0
    self.next_checkpoint_at = FULL_CHECKPOINT_INTERVAL
    self.last_model_save_transition = -1
    self.static_invalid_count = 0
    self.race_invalid_count = 0
    self.action_counts = {action: 0 for action in ACTIONS}
    self.reward_totals = {
        "official": 0.0,
        "crate": 0.0,
        "death": 0.0,
        "invalid": 0.0,
        "potential": 0.0,
        "total": 0.0,
    }
    self.round_train_return = 0.0
    self.round_official_return = 0.0
    self.metric_window = _fresh_metric_window()
    self.transfer_max_q_error = None

    MODEL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RESUME_TRAINING:
        _load_training_checkpoint(self)
        self.logger.info(
            "Resumed control run at transition %d and optimizer step %d",
            self.transition_count,
            self.optimizer_step,
        )
        return

    old_snapshots = sorted(MODEL_SNAPSHOT_DIR.glob("m2_2_control_t*.pt"))
    if old_snapshots:
        raise FileExistsError(
            "A fresh control run cannot reuse old snapshots: {}".format(
                MODEL_SNAPSHOT_DIR
            )
        )
    if TRAINING_CHECKPOINT_PATH.is_file():
        raise FileExistsError(
            "A fresh control run cannot overwrite this checkpoint: {}".format(
                TRAINING_CHECKPOINT_PATH
            )
        )

    with open(METRICS_PATH, "w", encoding="utf-8"):
        pass
    _migrate_m2_1_weights(self)
    self.target_net.load_state_dict(self.online_net.state_dict(), strict=True)
    self.target_net.eval()
    for parameter in self.target_net.parameters():
        parameter.requires_grad_(False)

    _write_metric(
        {
            "kind": "transfer",
            "source_model": str(M2_1_MODEL_PATH),
            "max_q_error": self.transfer_max_q_error,
            "optimizer": "new_adam",
            "replay": "fresh_lap",
            "experiment_role": "m2_2_matched_control",
        }
    )
    _save_model_snapshot(self)
    _save_evaluation_model(self)
    self.last_model_save_transition = 0
    self.logger.info("Control run started with max Q error %.3e", self.transfer_max_q_error)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list,
):
    if self.pending_transition is not None:
        _commit_transition(self, self.pending_transition, done=False)
    if old_game_state is None or new_game_state is None:
        raise ValueError("A surviving step must provide old and new game states")
    if self_action not in ACTION_TO_INDEX:
        raise ValueError("Unknown action returned by agent: " + repr(self_action))
    self.pending_transition = _make_pending(
        old_game_state,
        self_action,
        new_game_state,
        events,
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    last_key = _state_key(last_game_state)
    pending_matches_last_action = (
        self.pending_transition is not None
        and last_key is not None
        and (self.pending_transition.round_number, self.pending_transition.step_number)
        == last_key
        and self.pending_transition.action_name == last_action
    )

    if pending_matches_last_action:
        _commit_transition(
            self,
            self.pending_transition,
            done=True,
            terminal_events=tuple(events),
        )
    else:
        if self.pending_transition is not None:
            _commit_transition(self, self.pending_transition, done=False)
        if last_game_state is not None and last_action in ACTION_TO_INDEX:
            final_transition = _make_pending(
                last_game_state,
                last_action,
                None,
                events,
            )
            _commit_transition(self, final_transition, done=True)

    self.pending_transition = None
    if len(self.n_step_accumulator) != 0:
        raise RuntimeError("The n-step queue was not empty at the end of a round")
    if self.source_transition_count != self.transition_count:
        raise RuntimeError("Source and replay transition counts diverged")
    self.rounds_completed += 1

    _write_metric(
        {
            "kind": "round",
            "rounds_completed": self.rounds_completed,
            "decision_step": self.decision_step,
            "source_transition_count": self.source_transition_count,
            "transition_count": self.transition_count,
            "optimizer_step": self.optimizer_step,
            "train_return": self.round_train_return,
            "official_return": self.round_official_return,
            "static_invalid": self.static_invalid_count,
            "race_invalid": self.race_invalid_count,
        }
    )
    self.round_train_return = 0.0
    self.round_official_return = 0.0

    if self.last_model_save_transition != self.transition_count:
        _save_evaluation_model(self)
        self.last_model_save_transition = self.transition_count
    if (
        self.transition_count >= MAX_TRAINING_TRANSITIONS
        and self.metric_window["updates"] > 0
    ):
        _flush_training_metrics(self)
    if self.transition_count >= self.next_checkpoint_at:
        while self.next_checkpoint_at <= self.transition_count:
            self.next_checkpoint_at += FULL_CHECKPOINT_INTERVAL
        _save_training_checkpoint(self)


def _make_pending(old_game_state, action, new_game_state, events):
    round_number, step_number = _state_key(old_game_state)
    next_features = state_to_features(new_game_state)
    if new_game_state is None:
        next_mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    else:
        next_mask = legal_action_mask(new_game_state)
    return PendingTransition(
        state=state_to_features(old_game_state),
        action_index=ACTION_TO_INDEX[action],
        action_name=action,
        next_state=next_features,
        next_mask=next_mask,
        old_mask=legal_action_mask(old_game_state),
        round_number=round_number,
        step_number=step_number,
        events=tuple(events),
    )


def _state_key(game_state):
    if game_state is None:
        return None
    return int(game_state["round"]), int(game_state["step"])


def _commit_transition(self, transition, done, terminal_events=None):
    if self.source_transition_count >= MAX_TRAINING_TRANSITIONS:
        return
    used_events = transition.events if terminal_events is None else terminal_events
    reward, components = _reward_for_transition(self, transition, used_events, done)
    one_step = OneStepTransition(
        state=transition.state,
        action=transition.action_index,
        reward=reward,
        next_state=None if done else transition.next_state,
        done=done,
        next_mask=None if done else transition.next_mask,
    )

    ready = self.n_step_accumulator.append(one_step)
    self.source_transition_count += 1
    self.action_counts[transition.action_name] += 1
    for name, value in components.items():
        self.reward_totals[name] += value
    self.round_train_return += reward
    self.round_official_return += components["official"]

    if (
        self.source_transition_count == MAX_TRAINING_TRANSITIONS
        and len(self.n_step_accumulator) > 0
    ):
        ready.extend(self.n_step_accumulator.flush())
    for n_step_transition in ready:
        _store_and_learn(self, n_step_transition)

    if self.source_transition_count == MAX_TRAINING_TRANSITIONS:
        if self.transition_count != MAX_TRAINING_TRANSITIONS:
            raise RuntimeError("The final n-step flush changed the control budget")
        self.logger.info(
            "Reached the control budget of %d new transitions",
            self.transition_count,
        )


def _store_and_learn(self, transition):
    if self.transition_count >= MAX_TRAINING_TRANSITIONS:
        raise RuntimeError("Tried to store beyond the control budget")
    if not 1 <= int(transition.horizon) <= N_STEP:
        raise ValueError("An n-step sample has an invalid horizon")
    self.replay_buffer.add(
        transition.state,
        transition.action,
        transition.reward,
        transition.next_state,
        transition.done,
        transition.next_mask,
        transition.horizon,
    )
    self.transition_count += 1

    ready_to_learn = len(self.replay_buffer) >= WARMUP_TRANSITIONS
    on_replay_step = (
        self.transition_count - WARMUP_TRANSITIONS
    ) % REPLAY_PERIOD == 0
    if ready_to_learn and on_replay_step:
        _learn_once(self)
    if self.transition_count % MODEL_SNAPSHOT_INTERVAL == 0:
        _save_model_snapshot(self)


def _reward_for_transition(self, transition, events, done):
    official = events.count(e.COIN_COLLECTED) * REWARD_COIN
    official += events.count(e.KILLED_OPPONENT) * REWARD_KILL
    crate_reward = events.count(e.CRATE_DESTROYED) * REWARD_CRATE

    if e.KILLED_SELF in events:
        death_reward = PENALTY_SUICIDE
    elif e.GOT_KILLED in events:
        death_reward = PENALTY_KILLED
    else:
        death_reward = 0.0

    invalid_reward = 0.0
    if e.INVALID_ACTION in events:
        was_available = bool(transition.old_mask[transition.action_index])
        if was_available:
            self.race_invalid_count += 1
            invalid_reward = PENALTY_RACE_INVALID
        else:
            self.static_invalid_count += 1
            invalid_reward = PENALTY_STATIC_INVALID
            message = "Static invalid action at round {}, step {}: {}".format(
                transition.round_number,
                transition.step_number,
                transition.action_name,
            )
            if FAIL_ON_STATIC_INVALID:
                raise RuntimeError(message)
            self.logger.error(message)

    potential_reward = 0.0
    if USE_POTENTIAL_SHAPING:
        old_potential = potential_from_features(transition.state)
        next_potential = 0.0 if done else potential_from_features(transition.next_state)
        potential_reward = POTENTIAL_BETA * (GAMMA * next_potential - old_potential)

    total = official + crate_reward + death_reward + invalid_reward + potential_reward
    components = {
        "official": float(official),
        "crate": float(crate_reward),
        "death": float(death_reward),
        "invalid": float(invalid_reward),
        "potential": float(potential_reward),
        "total": float(total),
    }
    return float(total), components


def _learn_once(self):
    batch = self.replay_buffer.sample(BATCH_SIZE, self.rng)
    states = torch.from_numpy(batch["states"]).to(self.device)
    actions = torch.from_numpy(batch["actions"]).to(self.device)
    rewards = torch.from_numpy(batch["rewards"]).to(self.device)
    next_states = torch.from_numpy(batch["next_states"]).to(self.device)
    dones = torch.from_numpy(batch["dones"]).to(self.device)
    next_masks = torch.from_numpy(batch["next_masks"]).to(self.device)
    horizons = torch.from_numpy(batch["horizons"]).to(self.device)
    if torch.any(horizons < 1) or torch.any(horizons > N_STEP):
        raise ValueError("LAP replay returned an invalid n-step horizon")

    self.online_net.train()
    all_q_values = self.online_net(states)
    chosen_q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_values = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=self.device)
        non_terminal = ~dones
        if non_terminal.any():
            live_next_states = next_states[non_terminal]
            live_next_masks = next_masks[non_terminal]
            online_next_q = self.online_net(live_next_states)
            online_next_q = mask_q_values(online_next_q, live_next_masks)
            next_actions = online_next_q.argmax(dim=1)
            target_next_q = self.target_net(live_next_states)
            next_values[non_terminal] = target_next_q.gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
        discounts = torch.pow(
            torch.full_like(rewards, GAMMA),
            horizons.to(dtype=rewards.dtype),
        )
        targets = rewards + discounts * next_values

    if not torch.isfinite(chosen_q_values).all() or not torch.isfinite(targets).all():
        raise FloatingPointError("Non-finite Q-value or control target")
    td_errors = targets - chosen_q_values
    loss = F.smooth_l1_loss(chosen_q_values, targets, beta=1.0)
    if not torch.isfinite(loss) or not torch.isfinite(td_errors).all():
        raise FloatingPointError("Non-finite control training value")

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.online_net.parameters(),
        GRAD_CLIP_NORM,
    )
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Non-finite gradient norm")
    self.optimizer.step()
    self.optimizer_step += 1
    self.replay_buffer.update_from_td_errors(
        batch["indices"],
        td_errors.detach().cpu().numpy(),
    )
    if self.optimizer_step % TARGET_UPDATE_INTERVAL == 0:
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    window = self.metric_window
    window["updates"] += 1
    window["loss_sum"] += float(loss.item())
    window["td_abs_sum"] += float(td_errors.detach().abs().mean().item())
    window["td_abs_max"] = max(
        window["td_abs_max"],
        float(td_errors.detach().abs().max().item()),
    )
    window["grad_norm_sum"] += float(grad_norm.item())
    window["q_sum"] += float(chosen_q_values.detach().sum().item())
    window["q_count"] += int(chosen_q_values.numel())
    window["q_abs_max"] = max(
        window["q_abs_max"],
        float(chosen_q_values.detach().abs().max().item()),
    )
    window["sample_priority_sum"] += float(batch["raw_priorities"].mean())
    if self.optimizer_step % METRIC_LOG_INTERVAL == 0:
        _flush_training_metrics(self)


def _flush_training_metrics(self):
    window = self.metric_window
    updates = max(window["updates"], 1)
    q_count = max(window["q_count"], 1)
    priority_mean, priority_max, fraction_above_floor = (
        self.replay_buffer.priority_statistics()
    )
    data = {
        "kind": "training",
        "experiment_role": "m2_2_matched_control",
        "decision_step": self.decision_step,
        "source_transition_count": self.source_transition_count,
        "transition_count": self.transition_count,
        "optimizer_step": self.optimizer_step,
        "epsilon": epsilon_for_step(self.decision_step),
        "n_step": N_STEP,
        "replay": "loss_adjusted_prioritized",
        "replay_period": REPLAY_PERIOD,
        "loss_mean": window["loss_sum"] / updates,
        "td_abs_mean": window["td_abs_sum"] / updates,
        "td_abs_max": window["td_abs_max"],
        "grad_norm_mean": window["grad_norm_sum"] / updates,
        "q_mean": window["q_sum"] / q_count,
        "q_abs_max": window["q_abs_max"],
        "sample_priority_mean": window["sample_priority_sum"] / updates,
        "replay_priority_mean": priority_mean,
        "replay_priority_max": priority_max,
        "priority_above_floor_fraction": fraction_above_floor,
        "buffer_size": len(self.replay_buffer),
        "action_counts": dict(self.action_counts),
        "reward_totals": dict(self.reward_totals),
        "static_invalid": self.static_invalid_count,
        "race_invalid": self.race_invalid_count,
    }
    _write_metric(data)
    self.logger.info(
        "control_step=%d loss=%.4f td=%.4f epsilon=%.3f",
        self.optimizer_step,
        data["loss_mean"],
        data["td_abs_mean"],
        data["epsilon"],
    )
    self.metric_window = _fresh_metric_window()


def _migrate_m2_1_weights(self):
    if not M2_1_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Control run needs the frozen M2-1 model at {}".format(M2_1_MODEL_PATH)
        )
    checkpoint = _load_torch_file(M2_1_MODEL_PATH, self.device)
    metadata = checkpoint.get("metadata", {})
    required = {
        "variant_id": "M2-1",
        "algorithm_id": "dueling_double_dqn_3step_lap",
        "feature_schema_version": "1.4",
        "feature_dim": FEATURE_DIM,
        "actions": list(ACTIONS),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError("The M2-1 source checkpoint has the wrong " + key)
    if int(checkpoint.get("transition_count", -1)) != 500_000:
        raise ValueError("Control must start from the frozen 500k M2-1 checkpoint")

    source_state = checkpoint.get("model_state_dict")
    if source_state is None:
        raise ValueError("The M2-1 checkpoint has no model_state_dict")
    self.online_net.load_state_dict(source_state, strict=True)

    reference = DuelingDQN(input_dim=FEATURE_DIM, action_dim=len(ACTIONS)).to(self.device)
    reference.load_state_dict(source_state, strict=True)
    reference.eval()
    with torch.no_grad():
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=4 * FEATURE_DIM,
            device=self.device,
        ).reshape(4, FEATURE_DIM)
        max_error = float((reference(probe) - self.online_net(probe)).abs().max().item())
    if max_error > 1e-6:
        raise RuntimeError("Control migration changed M2-1 Q-values")
    self.transfer_max_q_error = max_error


def _write_metric(data):
    with open(METRICS_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, sort_keys=True) + "\n")


def _atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _evaluation_model_payload(self):
    return {
        "metadata": checkpoint_metadata(),
        "model_state_dict": self.online_net.state_dict(),
        "decision_step": self.decision_step,
        "source_transition_count": self.source_transition_count,
        "transition_count": self.transition_count,
        "optimizer_step": self.optimizer_step,
        "transfer_max_q_error": self.transfer_max_q_error,
    }


def _save_evaluation_model(self):
    _atomic_torch_save(_evaluation_model_payload(self), MODEL_PATH)


def _save_model_snapshot(self):
    snapshot_path = MODEL_SNAPSHOT_DIR / "m2_2_control_t{:07d}.pt".format(
        self.transition_count
    )
    if snapshot_path.exists():
        raise FileExistsError("Refusing to overwrite control snapshot: {}".format(snapshot_path))
    _atomic_torch_save(_evaluation_model_payload(self), snapshot_path)
    self.logger.info("Saved control validation snapshot at %d", self.transition_count)


def _save_training_checkpoint(self):
    if self.pending_transition is not None:
        raise RuntimeError("Checkpoint requested before pending state was cleared")
    if len(self.n_step_accumulator) != 0:
        raise RuntimeError("Checkpoint requested before the n-step queue was flushed")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": checkpoint_metadata(),
        "online_state_dict": self.online_net.state_dict(),
        "target_state_dict": self.target_net.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "replay_buffer": self.replay_buffer.state_dict(),
        "n_step_accumulator": self.n_step_accumulator.state_dict(),
        "decision_step": self.decision_step,
        "source_transition_count": self.source_transition_count,
        "transition_count": self.transition_count,
        "optimizer_step": self.optimizer_step,
        "rounds_completed": self.rounds_completed,
        "next_checkpoint_at": self.next_checkpoint_at,
        "last_model_save_transition": self.last_model_save_transition,
        "static_invalid_count": self.static_invalid_count,
        "race_invalid_count": self.race_invalid_count,
        "action_counts": dict(self.action_counts),
        "reward_totals": dict(self.reward_totals),
        "metric_window": dict(self.metric_window),
        "transfer_max_q_error": self.transfer_max_q_error,
        "numpy_rng_state": self.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    _atomic_torch_save(payload, TRAINING_CHECKPOINT_PATH)
    self.logger.info("Saved full control checkpoint at %d", self.transition_count)


def _load_training_checkpoint(self):
    if not TRAINING_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "RESUME_TRAINING is enabled but no checkpoint exists at {}".format(
                TRAINING_CHECKPOINT_PATH
            )
        )
    try:
        checkpoint = torch.load(
            TRAINING_CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(TRAINING_CHECKPOINT_PATH, map_location=self.device)

    metadata = checkpoint.get("metadata", {})
    if metadata.get("config_hash") != config_hash():
        raise ValueError("Control checkpoint configuration does not match")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported control checkpoint format")

    self.online_net.load_state_dict(checkpoint["online_state_dict"], strict=True)
    self.target_net.load_state_dict(checkpoint["target_state_dict"], strict=True)
    self.target_net.eval()
    for parameter in self.target_net.parameters():
        parameter.requires_grad_(False)
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    self.n_step_accumulator.load_state_dict(checkpoint["n_step_accumulator"])
    if len(self.n_step_accumulator) != 0:
        raise ValueError("A round-boundary checkpoint needs an empty n-step queue")

    self.decision_step = int(checkpoint["decision_step"])
    self.source_transition_count = int(checkpoint["source_transition_count"])
    self.transition_count = int(checkpoint["transition_count"])
    self.optimizer_step = int(checkpoint["optimizer_step"])
    self.rounds_completed = int(checkpoint["rounds_completed"])
    self.next_checkpoint_at = int(checkpoint["next_checkpoint_at"])
    self.last_model_save_transition = int(checkpoint["last_model_save_transition"])
    self.static_invalid_count = int(checkpoint["static_invalid_count"])
    self.race_invalid_count = int(checkpoint["race_invalid_count"])
    self.action_counts = dict(checkpoint["action_counts"])
    self.reward_totals = dict(checkpoint["reward_totals"])
    self.metric_window = dict(checkpoint["metric_window"])
    self.transfer_max_q_error = float(checkpoint["transfer_max_q_error"])
    self.rng.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"])
    self.pending_transition = None
