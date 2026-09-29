"""Training callbacks for the interleaved M2-3 combat curriculum."""

from dataclasses import dataclass
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import events as e

from .callbacks import epsilon_for_step
from .config import (
    ACTIONS,
    ACTION_TO_INDEX,
    ADAM_BETAS,
    ADAM_EPS,
    BASE_FEATURE_DIM,
    BATCH_SIZE,
    CHECKPOINT_AFTER_SESSION_ROUNDS,
    CHECKPOINT_FORMAT_VERSION,
    ESCAPE_IS_BETA,
    ESCAPE_LOSS_WARMUP_UPDATES,
    ESCAPE_LOSS_WEIGHT,
    FAIL_ON_STATIC_INVALID,
    FEATURE_DIM,
    FULL_CHECKPOINT_INTERVAL,
    GAMMA,
    GRAD_CLIP_NORM,
    LAP_ALPHA,
    LAP_PRIORITY_FLOOR,
    LEARNING_RATE,
    M2_2_MODEL_PATH,
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
    SESSION_TRANSITION_LIMIT,
    TARGET_UPDATE_INTERVAL,
    TRAINING_CHECKPOINT_PATH,
    TRANSFER_TOLERANCE,
    USE_POTENTIAL_SHAPING,
    VARIANT_ID,
    WARMUP_TRANSITIONS,
    checkpoint_metadata,
    config_hash,
)
from .features import potential_from_features, state_to_features
from .model import TemporalDuelingDQN, mask_q_values
from .n_step import NStepAccumulator, OneStepTransition
from .replay_buffer import LAPReplayBuffer


@dataclass
class PendingTransition:
    state: np.ndarray
    action_index: int
    action_name: str
    next_state: np.ndarray
    next_mask: np.ndarray
    old_mask: np.ndarray
    escape_labels: np.ndarray
    escape_loss_mask: np.ndarray
    round_number: int
    step_number: int
    events: tuple


def _fresh_metric_window():
    return {
        "updates": 0,
        "total_loss_sum": 0.0,
        "q_loss_sum": 0.0,
        "escape_loss_sum": 0.0,
        "td_abs_sum": 0.0,
        "td_abs_max": 0.0,
        "grad_norm_sum": 0.0,
        "q_sum": 0.0,
        "q_count": 0,
        "q_abs_max": 0.0,
        "sample_priority_sum": 0.0,
        "importance_weight_sum": 0.0,
        "escape_valid_count": 0,
        "escape_positive_count": 0,
        "escape_negative_count": 0,
        "escape_positive_correct": 0,
        "escape_negative_correct": 0,
        "bomb_label_count": 0,
        "bomb_positive_count": 0,
        "temporal_min": float("inf"),
        "temporal_max": float("-inf"),
    }


def setup_training(self):
    """Build fresh replay state and load the frozen M2-2 network."""
    self.target_net = TemporalDuelingDQN(
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
    self.session_rounds = 0
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
            "Resumed M2-3 at transition %d and optimizer step %d",
            self.transition_count,
            self.optimizer_step,
        )
        return

    old_snapshots = sorted(MODEL_SNAPSHOT_DIR.glob("m2_3_t*.pt"))
    if old_snapshots:
        raise FileExistsError(
            "A fresh M2-3 run cannot reuse snapshots from another run: {}".format(
                MODEL_SNAPSHOT_DIR
            )
        )
    if TRAINING_CHECKPOINT_PATH.is_file():
        raise FileExistsError(
            "A fresh M2-3 run cannot overwrite this checkpoint: {}".format(
                TRAINING_CHECKPOINT_PATH
            )
        )

    with open(METRICS_PATH, "w", encoding="utf-8"):
        pass
    _load_m2_2_weights(self)
    self.target_net.load_state_dict(self.online_net.state_dict(), strict=True)
    self.target_net.eval()
    for parameter in self.target_net.parameters():
        parameter.requires_grad_(False)

    _write_metric(
        {
            "kind": "transfer",
            "source_model": str(M2_2_MODEL_PATH),
            "max_q_error": self.transfer_max_q_error,
            "optimizer": "new_adam",
            "replay": "fresh_lap",
        }
    )
    _save_model_snapshot(self)
    _save_evaluation_model(self)
    self.last_model_save_transition = 0
    self.logger.info(
        "M2-3 started from M2-2 with max Q error %.3e",
        self.transfer_max_q_error,
    )


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list,
):
    """Delay one step until the next callback confirms its terminal flag."""
    if self.pending_transition is not None:
        _commit_transition(self, self.pending_transition, done=False)

    if old_game_state is None or new_game_state is None:
        raise ValueError("A surviving step must provide old and new game states")
    if self_action not in ACTION_TO_INDEX:
        raise ValueError("Unknown action returned by agent: " + repr(self_action))
    self.pending_transition = _make_pending(
        self,
        old_game_state,
        self_action,
        new_game_state,
        events,
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """Commit the last action, flush short n-step returns, and save safely."""
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
                self,
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
    self.session_rounds += 1

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
        self.transition_count >= SESSION_TRANSITION_LIMIT
        and self.metric_window["updates"] > 0
    ):
        _flush_training_metrics(self)
    if self.transition_count >= self.next_checkpoint_at:
        while self.next_checkpoint_at <= self.transition_count:
            self.next_checkpoint_at += FULL_CHECKPOINT_INTERVAL
        _save_training_checkpoint(self)
    elif (
        CHECKPOINT_AFTER_SESSION_ROUNDS > 0
        and self.session_rounds >= CHECKPOINT_AFTER_SESSION_ROUNDS
    ):
        _save_training_checkpoint(self)


def _analysis_for_old_state(self, game_state):
    # Calling the planner again is cheap here because the observation is cached.
    # The fingerprint separates pre-action and post-action states at the same step.
    return self.temporal_planner.analyze(game_state)


def _make_pending(self, old_game_state, action, new_game_state, events):
    round_number, step_number = _state_key(old_game_state)
    old_analysis = _analysis_for_old_state(self, old_game_state)
    state = state_to_features(old_game_state, old_analysis)
    if new_game_state is None:
        next_state = None
        next_mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    else:
        next_analysis = self.temporal_planner.analyze(new_game_state)
        next_state = state_to_features(new_game_state, next_analysis)
        next_mask = next_analysis.legal_mask

    escape_loss_mask = old_analysis.legal_mask.copy()
    if not old_analysis.known_hazard_present:
        escape_loss_mask[:5] = False
    return PendingTransition(
        state=state,
        action_index=ACTION_TO_INDEX[action],
        action_name=action,
        next_state=next_state,
        next_mask=next_mask,
        old_mask=old_analysis.legal_mask.copy(),
        escape_labels=old_analysis.escape_feasible.astype(np.float32),
        escape_loss_mask=escape_loss_mask,
        round_number=round_number,
        step_number=step_number,
        events=tuple(events),
    )


def _state_key(game_state):
    if game_state is None:
        return None
    return int(game_state["round"]), int(game_state["step"])


def _commit_transition(self, transition, done, terminal_events=None):
    if self.source_transition_count >= SESSION_TRANSITION_LIMIT:
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
        escape_labels=transition.escape_labels,
        escape_loss_mask=transition.escape_loss_mask,
    )

    ready = self.n_step_accumulator.append(one_step)
    self.source_transition_count += 1
    self.action_counts[transition.action_name] += 1
    for name, value in components.items():
        self.reward_totals[name] += value
    self.round_train_return += reward
    self.round_official_return += components["official"]

    temporal_values = transition.state[BASE_FEATURE_DIM:FEATURE_DIM]
    self.metric_window["temporal_min"] = min(
        self.metric_window["temporal_min"],
        float(temporal_values.min()),
    )
    self.metric_window["temporal_max"] = max(
        self.metric_window["temporal_max"],
        float(temporal_values.max()),
    )

    if (
        self.source_transition_count == SESSION_TRANSITION_LIMIT
        and len(self.n_step_accumulator) > 0
    ):
        ready.extend(self.n_step_accumulator.flush())
    for n_step_transition in ready:
        _store_and_learn(self, n_step_transition)

    if self.source_transition_count == SESSION_TRANSITION_LIMIT:
        if self.transition_count != SESSION_TRANSITION_LIMIT:
            raise RuntimeError("The final n-step flush changed the training budget")
        self.logger.info(
            "Reached the current M2-3 curriculum target of %d transitions",
            self.transition_count,
        )


def _store_and_learn(self, transition):
    if self.transition_count >= MAX_TRAINING_TRANSITIONS:
        raise RuntimeError("Tried to store a transition beyond the training budget")
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
        transition.escape_labels,
        transition.escape_loss_mask,
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


def _escape_weight(optimizer_step):
    progress = min(
        max(optimizer_step / float(ESCAPE_LOSS_WARMUP_UPDATES), 0.0),
        1.0,
    )
    return ESCAPE_LOSS_WEIGHT * progress


def _learn_once(self):
    batch = self.replay_buffer.sample(BATCH_SIZE, self.rng)
    states = torch.from_numpy(batch["states"]).to(self.device)
    actions = torch.from_numpy(batch["actions"]).to(self.device)
    rewards = torch.from_numpy(batch["rewards"]).to(self.device)
    next_states = torch.from_numpy(batch["next_states"]).to(self.device)
    dones = torch.from_numpy(batch["dones"]).to(self.device)
    next_masks = torch.from_numpy(batch["next_masks"]).to(self.device)
    horizons = torch.from_numpy(batch["horizons"]).to(self.device)
    escape_labels = torch.from_numpy(batch["escape_labels"]).to(self.device)
    escape_masks = torch.from_numpy(batch["escape_loss_masks"]).to(self.device)
    probabilities = torch.from_numpy(batch["sampling_probabilities"]).to(self.device)
    if torch.any(horizons < 1) or torch.any(horizons > N_STEP):
        raise ValueError("LAP replay returned an invalid n-step horizon")

    self.online_net.train()
    all_q_values, escape_logits = self.online_net.forward_with_escape(states)
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
        raise FloatingPointError("Non-finite Q-value or three-step target")
    td_errors = targets - chosen_q_values
    q_loss = F.smooth_l1_loss(chosen_q_values, targets, beta=1.0)

    importance_weights = torch.pow(
        len(self.replay_buffer) * probabilities,
        -ESCAPE_IS_BETA,
    )
    importance_weights = importance_weights / importance_weights.max().clamp_min(1e-8)
    element_loss = F.binary_cross_entropy_with_logits(
        escape_logits,
        escape_labels,
        reduction="none",
    )
    weighted_mask = escape_masks.to(dtype=torch.float32) * importance_weights.unsqueeze(1)
    valid_weight = weighted_mask.sum()
    if valid_weight.item() > 0.0:
        escape_loss = (element_loss * weighted_mask).sum() / valid_weight
    else:
        escape_loss = escape_logits.sum() * 0.0

    auxiliary_weight = _escape_weight(self.optimizer_step)
    total_loss = q_loss + auxiliary_weight * escape_loss
    finite_values = (q_loss, escape_loss, total_loss)
    if not all(torch.isfinite(value) for value in finite_values):
        raise FloatingPointError("Non-finite M2-3 training loss")
    if not torch.isfinite(td_errors).all():
        raise FloatingPointError("Non-finite LAP TD error")

    self.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
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

    _update_metric_window(
        self,
        batch,
        q_loss,
        escape_loss,
        total_loss,
        td_errors,
        chosen_q_values,
        escape_logits,
        escape_labels,
        escape_masks,
        importance_weights,
        grad_norm,
    )
    if self.optimizer_step % METRIC_LOG_INTERVAL == 0:
        _flush_training_metrics(self)


def _update_metric_window(
    self,
    batch,
    q_loss,
    escape_loss,
    total_loss,
    td_errors,
    chosen_q_values,
    escape_logits,
    escape_labels,
    escape_masks,
    importance_weights,
    grad_norm,
):
    window = self.metric_window
    window["updates"] += 1
    window["total_loss_sum"] += float(total_loss.item())
    window["q_loss_sum"] += float(q_loss.item())
    window["escape_loss_sum"] += float(escape_loss.item())
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
    window["importance_weight_sum"] += float(importance_weights.mean().item())

    with torch.no_grad():
        valid = escape_masks
        predictions = escape_logits >= 0.0
        positives = valid & (escape_labels > 0.5)
        negatives = valid & (escape_labels <= 0.5)
        window["escape_valid_count"] += int(valid.sum().item())
        window["escape_positive_count"] += int(positives.sum().item())
        window["escape_negative_count"] += int(negatives.sum().item())
        window["escape_positive_correct"] += int((predictions & positives).sum().item())
        window["escape_negative_correct"] += int((~predictions & negatives).sum().item())
        bomb_valid = valid[:, 5]
        window["bomb_label_count"] += int(bomb_valid.sum().item())
        window["bomb_positive_count"] += int(
            (bomb_valid & (escape_labels[:, 5] > 0.5)).sum().item()
        )


def _balanced_accuracy(window):
    recalls = []
    if window["escape_positive_count"] > 0:
        recalls.append(
            window["escape_positive_correct"] / window["escape_positive_count"]
        )
    if window["escape_negative_count"] > 0:
        recalls.append(
            window["escape_negative_correct"] / window["escape_negative_count"]
        )
    return sum(recalls) / len(recalls) if recalls else 0.0


def _flush_training_metrics(self):
    window = self.metric_window
    updates = max(window["updates"], 1)
    q_count = max(window["q_count"], 1)
    valid_count = max(window["escape_valid_count"], 1)
    bomb_count = max(window["bomb_label_count"], 1)
    priority_mean, priority_max, fraction_above_floor = (
        self.replay_buffer.priority_statistics()
    )
    shield = self.shield_window
    legal_total = max(shield["legal_actions"], 1)
    bomb_total = max(shield["bomb_candidates"], 1)
    temporal_min = window["temporal_min"]
    temporal_max = window["temporal_max"]
    if not np.isfinite(temporal_min):
        temporal_min = 0.0
    if not np.isfinite(temporal_max):
        temporal_max = 0.0

    data = {
        "kind": "training",
        "decision_step": self.decision_step,
        "source_transition_count": self.source_transition_count,
        "transition_count": self.transition_count,
        "optimizer_step": self.optimizer_step,
        "curriculum_block": os.environ.get(
            "SEAT_M2_3_CURRICULUM_BLOCK",
            "unspecified",
        ),
        "epsilon": epsilon_for_step(self.decision_step),
        "escape_lambda": _escape_weight(self.optimizer_step),
        "n_step": N_STEP,
        "replay": "loss_adjusted_prioritized",
        "replay_period": REPLAY_PERIOD,
        "loss_mean": window["total_loss_sum"] / updates,
        "q_loss_mean": window["q_loss_sum"] / updates,
        "escape_loss_mean": window["escape_loss_sum"] / updates,
        "escape_balanced_accuracy": _balanced_accuracy(window),
        "escape_positive_fraction": window["escape_positive_count"] / valid_count,
        "bomb_escape_positive_fraction": window["bomb_positive_count"] / bomb_count,
        "importance_weight_mean": window["importance_weight_sum"] / updates,
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
        "shield_random_decisions": shield["random_decisions"],
        "shield_prune_fraction": shield["pruned_actions"] / legal_total,
        "shield_bomb_prune_fraction": shield["bomb_pruned"] / bomb_total,
        "shield_no_safe_fallbacks": shield["no_safe_fallbacks"],
        "planner_requests": self.temporal_planner.requests,
        "planner_computations": self.temporal_planner.computations,
        "planner_cache_hits": self.temporal_planner.cache_hits,
        "temporal_feature_min": temporal_min,
        "temporal_feature_max": temporal_max,
        "action_counts": dict(self.action_counts),
        "reward_totals": dict(self.reward_totals),
        "static_invalid": self.static_invalid_count,
        "race_invalid": self.race_invalid_count,
    }
    _write_metric(data)
    self.logger.info(
        "step=%d q_loss=%.4f escape_loss=%.4f epsilon=%.3f",
        self.optimizer_step,
        data["q_loss_mean"],
        data["escape_loss_mean"],
        data["epsilon"],
    )
    self.metric_window = _fresh_metric_window()
    self.shield_window = {
        "random_decisions": 0,
        "legal_actions": 0,
        "pruned_actions": 0,
        "bomb_candidates": 0,
        "bomb_pruned": 0,
        "no_safe_fallbacks": 0,
    }


def _load_m2_2_weights(self):
    if not M2_2_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "M2-3 needs the frozen M2-2 model at {}".format(M2_2_MODEL_PATH)
        )
    checkpoint = _load_torch_file(M2_2_MODEL_PATH, self.device)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("variant_id") != "M2-2":
        raise ValueError("The warm-start checkpoint is not an M2-2 model")
    if metadata.get("algorithm_id") != "warm_started_3step_lap_temporal_escape":
        raise ValueError("The warm-start checkpoint uses the wrong M2-2 algorithm")
    if metadata.get("feature_schema_version") != "2.0":
        raise ValueError("The warm-start checkpoint uses the wrong feature schema")
    if metadata.get("feature_dim") != FEATURE_DIM:
        raise ValueError("The M2-2 checkpoint has an unexpected feature dimension")
    if metadata.get("actions") != list(ACTIONS):
        raise ValueError("The M2-2 checkpoint uses a different action order")
    if int(checkpoint.get("transition_count", -1)) != 250_000:
        raise ValueError("M2-3 must start from the frozen 250k M2-2 checkpoint")
    source_state = checkpoint.get("model_state_dict")
    if source_state is None:
        raise ValueError("The M2-2 checkpoint has no model_state_dict")

    self.online_net.load_state_dict(source_state, strict=True)
    with torch.no_grad():
        reference = TemporalDuelingDQN(FEATURE_DIM, len(ACTIONS)).to(self.device)
        reference.load_state_dict(source_state, strict=True)
        probe = torch.linspace(
            -1.0,
            1.0,
            steps=4 * FEATURE_DIM,
            device=self.device,
        ).reshape(4, FEATURE_DIM)
        max_error = float((reference(probe) - self.online_net(probe)).abs().max().item())
    if max_error > TRANSFER_TOLERANCE:
        raise RuntimeError(
            "M2-2 transfer changed Q-values by {:.3e}".format(max_error)
        )
    self.transfer_max_q_error = max_error


def _load_torch_file(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    safe_variant = VARIANT_ID.lower().replace("-", "_")
    snapshot_path = MODEL_SNAPSHOT_DIR / "{}_t{:07d}.pt".format(
        safe_variant,
        self.transition_count,
    )
    if snapshot_path.exists():
        raise FileExistsError("Refusing to overwrite model snapshot: {}".format(snapshot_path))
    _atomic_torch_save(_evaluation_model_payload(self), snapshot_path)
    self.logger.info("Saved M2-3 snapshot at %d", self.transition_count)


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
        "shield_window": dict(self.shield_window),
        "transfer_max_q_error": self.transfer_max_q_error,
        "planner_requests": self.temporal_planner.requests,
        "planner_computations": self.temporal_planner.computations,
        "planner_cache_hits": self.temporal_planner.cache_hits,
        "numpy_rng_state": self.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    _atomic_torch_save(payload, TRAINING_CHECKPOINT_PATH)
    self.logger.info("Saved full M2-3 checkpoint at %d", self.transition_count)


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
        raise ValueError("Training checkpoint configuration does not match this run")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported M2-3 training checkpoint format")

    self.online_net.load_state_dict(checkpoint["online_state_dict"], strict=True)
    self.target_net.load_state_dict(checkpoint["target_state_dict"], strict=True)
    self.target_net.eval()
    for parameter in self.target_net.parameters():
        parameter.requires_grad_(False)
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    self.n_step_accumulator.load_state_dict(checkpoint["n_step_accumulator"])
    if len(self.n_step_accumulator) != 0:
        raise ValueError("A round-boundary checkpoint must have an empty n-step queue")

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
    self.shield_window = dict(checkpoint["shield_window"])
    self.transfer_max_q_error = float(checkpoint["transfer_max_q_error"])
    self.temporal_planner.requests = int(checkpoint["planner_requests"])
    self.temporal_planner.computations = int(checkpoint["planner_computations"])
    self.temporal_planner.cache_hits = int(checkpoint["planner_cache_hits"])
    self.rng.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"])
    self.pending_transition = None
