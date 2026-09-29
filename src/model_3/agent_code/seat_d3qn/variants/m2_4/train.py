"""Training callbacks for the M2-4 residual experiments."""

from dataclasses import dataclass
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import events as e

from ..m2_3.features import potential_from_features, state_to_features
from ..m2_3.n_step import NStepAccumulator, OneStepTransition
from ..m2_3.replay_buffer import LAPReplayBuffer
from .callbacks import epsilon_for_step
from .config import (
    ACTIONS,
    ACTION_TO_INDEX,
    ADAM_BETAS,
    ADAM_EPS,
    ANCHOR_WEIGHT,
    BASE_FEATURE_DIM,
    BATCH_SIZE,
    BRANCH,
    C1_MODEL_PATH,
    C3_KEEP_WEIGHT,
    CHECKPOINT_AFTER_SESSION_ROUNDS,
    CHECKPOINT_FORMAT_VERSION,
    FAIL_ON_STATIC_INVALID,
    FEATURE_DIM,
    FULL_CHECKPOINT_INTERVAL,
    GAMMA,
    GRAD_CLIP_NORM,
    LAP_ALPHA,
    LAP_PRIORITY_FLOOR,
    LEARNING_RATE,
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
    TEACHER_BATCH_SIZE,
    TEACHER_BUFFER_CAPACITY,
    TEACHER_ENABLED,
    TEACHER_MARGIN,
    TRAINING_CHECKPOINT_PATH,
    TRANSFER_TOLERANCE,
    USE_POTENTIAL_SHAPING,
    VARIANT_ID,
    WARMUP_TRANSITIONS,
    checkpoint_metadata,
    config_hash,
    learning_rate,
    teacher_weight,
)
from .model import ResidualQNetwork, mask_q_values
from .teacher_buffer import TeacherReplayBuffer


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
    teacher_record: dict
    round_number: int
    step_number: int
    events: tuple


def _fresh_metric_window():
    return {
        "updates": 0,
        "total_loss_sum": 0.0,
        "q_loss_sum": 0.0,
        "teacher_loss_sum": 0.0,
        "anchor_loss_sum": 0.0,
        "teacher_loss_ratio_sum": 0.0,
        "teacher_batches": 0,
        "td_abs_sum": 0.0,
        "td_abs_max": 0.0,
        "grad_norm_sum": 0.0,
        "q_sum": 0.0,
        "q_count": 0,
        "q_abs_max": 0.0,
        "base_square_sum": 0.0,
        "residual_square_sum": 0.0,
        "value_count": 0,
        "sample_priority_sum": 0.0,
        "temporal_min": float("inf"),
        "temporal_max": float("-inf"),
    }


def setup_training(self):
    """Create fresh optimization state around the frozen M2-3 base."""
    self.target_residual_net = ResidualQNetwork(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.optimizer = torch.optim.Adam(
        self.residual_net.parameters(),
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
    self.teacher_buffer = TeacherReplayBuffer(TEACHER_BUFFER_CAPACITY, FEATURE_DIM)
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
    self.c1_source_sha256 = None
    self.reference_residual_net = None

    MODEL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if BRANCH == "c3":
        _load_c1_residual(self)

    if RESUME_TRAINING:
        _load_training_checkpoint(self)
        self.logger.info(
            "Resumed M2-4 %s at transition %d", BRANCH, self.transition_count
        )
        return

    prefix = "m2_4_{}_t".format(BRANCH)
    old_snapshots = sorted(MODEL_SNAPSHOT_DIR.glob(prefix + "*.pt"))
    if old_snapshots:
        raise FileExistsError(
            "A fresh M2-4 run cannot reuse snapshots in {}".format(MODEL_SNAPSHOT_DIR)
        )
    if TRAINING_CHECKPOINT_PATH.is_file():
        raise FileExistsError(
            "A fresh M2-4 run cannot overwrite {}".format(TRAINING_CHECKPOINT_PATH)
        )
    with open(METRICS_PATH, "w", encoding="utf-8"):
        pass

    if self.transfer_max_q_error > TRANSFER_TOLERANCE:
        raise RuntimeError(
            "Transferred residual changed the source Q-values by {:.3e}".format(
                self.transfer_max_q_error
            )
        )
    self.target_residual_net.load_state_dict(self.residual_net.state_dict(), strict=True)
    self.target_residual_net.eval()
    for parameter in self.target_residual_net.parameters():
        parameter.requires_grad_(False)

    _write_metric(
        {
            "kind": "transfer",
            "branch": BRANCH,
            "source_model_sha256": self.source_model_sha256,
            "c1_source_sha256": self.c1_source_sha256,
            "source_branch": "c1" if BRANCH == "c3" else "m2_3",
            "max_q_error": self.transfer_max_q_error,
            "optimizer": "new_adam",
            "replay": "fresh_lap",
            "teacher_replay": "fresh_separate_buffer" if TEACHER_ENABLED else "disabled",
        }
    )
    _save_model_snapshot(self)
    _save_evaluation_model(self)
    self.last_model_save_transition = 0


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
        self, old_game_state, self_action, new_game_state, events
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    last_key = _state_key(last_game_state)
    pending_matches = (
        self.pending_transition is not None
        and last_key is not None
        and (self.pending_transition.round_number, self.pending_transition.step_number)
        == last_key
        and self.pending_transition.action_name == last_action
    )
    if pending_matches:
        _commit_transition(
            self, self.pending_transition, done=True, terminal_events=tuple(events)
        )
    else:
        if self.pending_transition is not None:
            _commit_transition(self, self.pending_transition, done=False)
        if last_game_state is not None and last_action in ACTION_TO_INDEX:
            final_transition = _make_pending(
                self, last_game_state, last_action, None, events
            )
            _commit_transition(self, final_transition, done=True)

    self.pending_transition = None
    self.teacher_records.clear()
    if len(self.n_step_accumulator) != 0:
        raise RuntimeError("The n-step queue was not empty at the end of a round")
    if self.source_transition_count != self.transition_count:
        raise RuntimeError("Source and replay transition counts diverged")
    self.rounds_completed += 1
    self.session_rounds += 1
    _write_metric(
        {
            "kind": "round",
            "branch": BRANCH,
            "rounds_completed": self.rounds_completed,
            "decision_step": self.decision_step,
            "source_transition_count": self.source_transition_count,
            "transition_count": self.transition_count,
            "optimizer_step": self.optimizer_step,
            "train_return": self.round_train_return,
            "official_return": self.round_official_return,
            "static_invalid": self.static_invalid_count,
            "race_invalid": self.race_invalid_count,
            "teacher_queries": self.teacher_window["queries"],
            "teacher_accepted": self.teacher_window["accepted"],
            "teacher_low_confidence": self.teacher_window["low_confidence"],
            "teacher_illegal": self.teacher_window["illegal"],
            "teacher_agreement": self.teacher_window["agreement"],
            "teacher_irrelevant": self.teacher_window["irrelevant"],
            "teacher_unsafe": self.teacher_window["unsafe"],
        }
    )
    self.round_train_return = 0.0
    self.round_official_return = 0.0

    if self.last_model_save_transition != self.transition_count:
        _save_evaluation_model(self)
        self.last_model_save_transition = self.transition_count
    if self.transition_count >= SESSION_TRANSITION_LIMIT and self.metric_window["updates"]:
        _flush_training_metrics(self)
    if self.transition_count >= self.next_checkpoint_at:
        while self.next_checkpoint_at <= self.transition_count:
            self.next_checkpoint_at += FULL_CHECKPOINT_INTERVAL
        _save_training_checkpoint(self)
    elif CHECKPOINT_AFTER_SESSION_ROUNDS and self.session_rounds >= CHECKPOINT_AFTER_SESSION_ROUNDS:
        _save_training_checkpoint(self)


def _state_key(game_state):
    if game_state is None:
        return None
    return int(game_state["round"]), int(game_state["step"])


def _make_pending(self, old_game_state, action, new_game_state, events):
    round_number, step_number = _state_key(old_game_state)
    old_analysis = self.temporal_planner.analyze(old_game_state)
    state = state_to_features(old_game_state, old_analysis)
    if new_game_state is None:
        next_state = None
        next_mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    else:
        next_analysis = self.temporal_planner.analyze(new_game_state)
        next_state = state_to_features(new_game_state, next_analysis)
        next_mask = next_analysis.legal_mask.copy()

    record = self.teacher_records.pop((round_number, step_number), None)
    if TEACHER_ENABLED and record is None:
        raise RuntimeError("The C2 action has no matching teacher query record")
    escape_mask = old_analysis.legal_mask.copy()
    if not old_analysis.known_hazard_present:
        escape_mask[:5] = False
    return PendingTransition(
        state=state,
        action_index=ACTION_TO_INDEX[action],
        action_name=action,
        next_state=next_state,
        next_mask=next_mask,
        old_mask=old_analysis.legal_mask.copy(),
        escape_labels=old_analysis.escape_feasible.astype(np.float32),
        escape_loss_mask=escape_mask,
        teacher_record=record,
        round_number=round_number,
        step_number=step_number,
        events=tuple(events),
    )


def _commit_transition(self, transition, done, terminal_events=None):
    if self.source_transition_count >= SESSION_TRANSITION_LIMIT:
        return
    used_events = transition.events if terminal_events is None else terminal_events
    reward, components = _reward_for_transition(self, transition, used_events, done)

    if TEACHER_ENABLED and transition.teacher_record["accepted"]:
        self.teacher_buffer.add(
            transition.state,
            transition.old_mask,
            transition.teacher_record["action"],
            transition.teacher_record["confidence"],
        )

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

    temporal = transition.state[BASE_FEATURE_DIM:FEATURE_DIM]
    self.metric_window["temporal_min"] = min(
        self.metric_window["temporal_min"], float(temporal.min())
    )
    self.metric_window["temporal_max"] = max(
        self.metric_window["temporal_max"], float(temporal.max())
    )

    if self.source_transition_count == SESSION_TRANSITION_LIMIT and len(self.n_step_accumulator):
        ready.extend(self.n_step_accumulator.flush())
    for item in ready:
        _store_and_learn(self, item)
    if self.source_transition_count == SESSION_TRANSITION_LIMIT:
        if self.transition_count != SESSION_TRANSITION_LIMIT:
            raise RuntimeError("The final n-step flush changed the M2-4 budget")
        self.logger.info("Reached M2-4 %s target %d", BRANCH, self.transition_count)


def _store_and_learn(self, transition):
    if self.transition_count >= MAX_TRAINING_TRANSITIONS:
        raise RuntimeError("Tried to store a transition beyond the M2-4 budget")
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
    ready = len(self.replay_buffer) >= WARMUP_TRANSITIONS
    replay_step = (self.transition_count - WARMUP_TRANSITIONS) % REPLAY_PERIOD == 0
    if ready and replay_step:
        _learn_once(self)
    if self.transition_count % MODEL_SNAPSHOT_INTERVAL == 0:
        _save_model_snapshot(self)


def _reward_for_transition(self, transition, events, done):
    official = events.count(e.COIN_COLLECTED) * REWARD_COIN
    official += events.count(e.KILLED_OPPONENT) * REWARD_KILL
    crate = events.count(e.CRATE_DESTROYED) * REWARD_CRATE
    if e.KILLED_SELF in events:
        death = PENALTY_SUICIDE
    elif e.GOT_KILLED in events:
        death = PENALTY_KILLED
    else:
        death = 0.0

    invalid = 0.0
    if e.INVALID_ACTION in events:
        was_available = bool(transition.old_mask[transition.action_index])
        if was_available:
            self.race_invalid_count += 1
            invalid = PENALTY_RACE_INVALID
        else:
            self.static_invalid_count += 1
            invalid = PENALTY_STATIC_INVALID
            message = "Static invalid action at round {}, step {}: {}".format(
                transition.round_number, transition.step_number, transition.action_name
            )
            if FAIL_ON_STATIC_INVALID:
                raise RuntimeError(message)
            self.logger.error(message)

    potential = 0.0
    if USE_POTENTIAL_SHAPING:
        old_potential = potential_from_features(transition.state)
        next_potential = 0.0 if done else potential_from_features(transition.next_state)
        potential = POTENTIAL_BETA * (GAMMA * next_potential - old_potential)
    total = official + crate + death + invalid + potential
    return float(total), {
        "official": float(official),
        "crate": float(crate),
        "death": float(death),
        "invalid": float(invalid),
        "potential": float(potential),
        "total": float(total),
    }


def _learn_once(self):
    current_learning_rate = learning_rate(self.transition_count)
    for group in self.optimizer.param_groups:
        group["lr"] = current_learning_rate
    batch = self.replay_buffer.sample(BATCH_SIZE, self.rng)
    states = torch.from_numpy(batch["states"]).to(self.device)
    actions = torch.from_numpy(batch["actions"]).to(self.device)
    rewards = torch.from_numpy(batch["rewards"]).to(self.device)
    next_states = torch.from_numpy(batch["next_states"]).to(self.device)
    dones = torch.from_numpy(batch["dones"]).to(self.device)
    next_masks = torch.from_numpy(batch["next_masks"]).to(self.device)
    horizons = torch.from_numpy(batch["horizons"]).to(self.device)

    self.residual_net.train()
    residual_values = self.residual_net(states)
    with torch.no_grad():
        base_values = self.base_net(states)
    total_values = base_values + residual_values
    chosen_values = total_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_values = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=self.device)
        live = ~dones
        if live.any():
            live_states = next_states[live]
            live_masks = next_masks[live]
            base_next = self.base_net(live_states)
            online_next = base_next + self.residual_net(live_states)
            next_actions = mask_q_values(online_next, live_masks).argmax(dim=1)
            target_next = base_next + self.target_residual_net(live_states)
            next_values[live] = target_next.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
        discounts = torch.pow(
            torch.full_like(rewards, GAMMA), horizons.to(dtype=rewards.dtype)
        )
        targets = rewards + discounts * next_values

    td_errors = targets - chosen_values
    q_loss = F.smooth_l1_loss(chosen_values, targets, beta=1.0)
    if BRANCH == "c3":
        with torch.no_grad():
            reference_values = self.reference_residual_net(states)
        anchor_loss = (residual_values - reference_values).square().mean()
        anchor_weight = C3_KEEP_WEIGHT
    else:
        anchor_loss = residual_values.square().mean()
        anchor_weight = ANCHOR_WEIGHT
    teacher_loss = residual_values.sum() * 0.0
    teacher_batch_used = False

    if TEACHER_ENABLED and len(self.teacher_buffer) >= TEACHER_BATCH_SIZE:
        labels = self.teacher_buffer.sample(TEACHER_BATCH_SIZE, self.rng)
        label_states = torch.from_numpy(labels["states"]).to(self.device)
        label_masks = torch.from_numpy(labels["legal_masks"]).to(self.device)
        label_actions = torch.from_numpy(labels["actions"]).to(self.device)
        confidences = torch.from_numpy(labels["confidences"]).to(self.device)
        with torch.no_grad():
            teacher_base = self.base_net(label_states)
        teacher_q = teacher_base + self.residual_net(label_states)
        margins = torch.full_like(teacher_q, TEACHER_MARGIN)
        margins.scatter_(1, label_actions.unsqueeze(1), 0.0)
        ranked = mask_q_values(teacher_q + margins, label_masks)
        largest = ranked.max(dim=1).values
        teacher_values = teacher_q.gather(1, label_actions.unsqueeze(1)).squeeze(1)
        teacher_loss = (confidences * (largest - teacher_values)).mean()
        teacher_batch_used = True

    current_teacher_weight = teacher_weight(self.transition_count)
    total_loss = (
        q_loss
        + anchor_weight * anchor_loss
        + current_teacher_weight * teacher_loss
    )
    if not all(torch.isfinite(value) for value in (q_loss, anchor_loss, teacher_loss, total_loss)):
        raise FloatingPointError("Non-finite M2-4 loss")

    self.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.residual_net.parameters(), GRAD_CLIP_NORM
    )
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Non-finite M2-4 gradient norm")
    self.optimizer.step()
    self.optimizer_step += 1

    self.replay_buffer.update_from_td_errors(
        batch["indices"], td_errors.detach().cpu().numpy()
    )
    if self.optimizer_step % TARGET_UPDATE_INTERVAL == 0:
        self.target_residual_net.load_state_dict(
            self.residual_net.state_dict(), strict=True
        )
        self.target_residual_net.eval()

    _update_metric_window(
        self,
        batch,
        q_loss,
        teacher_loss,
        anchor_loss,
        total_loss,
        current_teacher_weight,
        teacher_batch_used,
        td_errors,
        chosen_values,
        base_values,
        residual_values,
        grad_norm,
    )
    if self.optimizer_step % METRIC_LOG_INTERVAL == 0:
        _flush_training_metrics(self)


def _update_metric_window(
    self,
    batch,
    q_loss,
    teacher_loss,
    anchor_loss,
    total_loss,
    current_teacher_weight,
    teacher_batch_used,
    td_errors,
    chosen_values,
    base_values,
    residual_values,
    grad_norm,
):
    window = self.metric_window
    window["updates"] += 1
    window["total_loss_sum"] += float(total_loss.item())
    window["q_loss_sum"] += float(q_loss.item())
    window["anchor_loss_sum"] += float(anchor_loss.item())
    if teacher_batch_used:
        window["teacher_batches"] += 1
        window["teacher_loss_sum"] += float(teacher_loss.item())
        ratio = current_teacher_weight * float(teacher_loss.item()) / max(
            float(q_loss.item()), 1e-8
        )
        window["teacher_loss_ratio_sum"] += ratio
    window["td_abs_sum"] += float(td_errors.detach().abs().mean().item())
    window["td_abs_max"] = max(
        window["td_abs_max"], float(td_errors.detach().abs().max().item())
    )
    window["grad_norm_sum"] += float(grad_norm.item())
    window["q_sum"] += float(chosen_values.detach().sum().item())
    window["q_count"] += int(chosen_values.numel())
    window["q_abs_max"] = max(
        window["q_abs_max"], float(chosen_values.detach().abs().max().item())
    )
    window["base_square_sum"] += float(base_values.detach().square().sum().item())
    window["residual_square_sum"] += float(
        residual_values.detach().square().sum().item()
    )
    window["value_count"] += int(base_values.numel())
    window["sample_priority_sum"] += float(batch["raw_priorities"].mean())


def _flush_training_metrics(self):
    window = self.metric_window
    updates = max(window["updates"], 1)
    q_count = max(window["q_count"], 1)
    value_count = max(window["value_count"], 1)
    teacher_batches = max(window["teacher_batches"], 1)
    priority_mean, priority_max, fraction_above_floor = (
        self.replay_buffer.priority_statistics()
    )
    base_rms = (window["base_square_sum"] / value_count) ** 0.5
    residual_rms = (window["residual_square_sum"] / value_count) ** 0.5
    residual_ratio = residual_rms / max(base_rms, 1e-8)
    temporal_min = window["temporal_min"]
    temporal_max = window["temporal_max"]
    if not np.isfinite(temporal_min):
        temporal_min = 0.0
    if not np.isfinite(temporal_max):
        temporal_max = 0.0

    teacher = self.teacher_window
    queries = max(teacher["queries"], 1)
    bomb_labels = max(teacher["bomb_labels"], 1)
    shield = self.shield_window
    legal_total = max(shield["legal_actions"], 1)
    bomb_total = max(shield["bomb_candidates"], 1)
    data = {
        "kind": "training",
        "branch": BRANCH,
        "decision_step": self.decision_step,
        "source_transition_count": self.source_transition_count,
        "transition_count": self.transition_count,
        "optimizer_step": self.optimizer_step,
        "curriculum_block": os.environ.get("SEAT_M2_4_CURRICULUM_BLOCK", "unspecified"),
        "epsilon": epsilon_for_step(self.decision_step),
        "learning_rate": learning_rate(self.transition_count),
        "teacher_lambda": teacher_weight(self.transition_count),
        "anchor_target": "selected_c1" if BRANCH == "c3" else "zero_residual",
        "loss_mean": window["total_loss_sum"] / updates,
        "q_loss_mean": window["q_loss_sum"] / updates,
        "teacher_loss_mean": window["teacher_loss_sum"] / teacher_batches,
        "teacher_to_td_loss_ratio": window["teacher_loss_ratio_sum"] / teacher_batches,
        "anchor_loss_mean": window["anchor_loss_sum"] / updates,
        "td_abs_mean": window["td_abs_sum"] / updates,
        "td_abs_max": window["td_abs_max"],
        "grad_norm_mean": window["grad_norm_sum"] / updates,
        "q_mean": window["q_sum"] / q_count,
        "q_abs_max": window["q_abs_max"],
        "base_q_rms": base_rms,
        "residual_q_rms": residual_rms,
        "residual_to_base_rms": residual_ratio,
        "teacher_queries": teacher["queries"],
        "teacher_accepted": teacher["accepted"],
        "teacher_label_coverage": teacher["accepted"] / queries,
        "teacher_confidence_mean": teacher["confidence_sum"] / queries,
        "teacher_low_confidence": teacher["low_confidence"],
        "teacher_illegal": teacher["illegal"],
        "teacher_agreement": teacher["agreement"],
        "teacher_irrelevant": teacher["irrelevant"],
        "teacher_unsafe": teacher["unsafe"],
        "teacher_bomb_labels": teacher["bomb_labels"],
        "teacher_direct_use_fraction": teacher["useful_bomb_labels"] / bomb_labels,
        "teacher_buffer_size": len(self.teacher_buffer),
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
    self.metric_window = _fresh_metric_window()
    from .callbacks import _fresh_shield_window, _fresh_teacher_window

    self.shield_window = _fresh_shield_window()
    self.teacher_window = _fresh_teacher_window()


def _write_metric(data):
    with open(METRICS_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, sort_keys=True) + "\n")


def _load_c1_residual(self):
    if not C1_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "C3 needs the selected C1 model at {}".format(C1_MODEL_PATH)
        )
    try:
        checkpoint = torch.load(C1_MODEL_PATH, map_location=self.device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(C1_MODEL_PATH, map_location=self.device)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("variant_id") != "M2-4-C1":
        raise ValueError("The C3 source is not a selected C1 model")
    if metadata.get("source_model_sha256") != self.source_model_sha256:
        raise ValueError("C1 and C3 do not share the same frozen M2-3 source")
    source_state = checkpoint.get("residual_state_dict")
    if source_state is None:
        raise ValueError("The selected C1 model has no residual weights")
    self.c1_source_sha256 = _file_sha256(C1_MODEL_PATH)
    self.residual_net.load_state_dict(source_state, strict=True)
    self.reference_residual_net = ResidualQNetwork(
        FEATURE_DIM, len(ACTIONS)
    ).to(self.device)
    self.reference_residual_net.load_state_dict(source_state, strict=True)
    self.reference_residual_net.eval()
    for parameter in self.reference_residual_net.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        probe = torch.linspace(-1.0, 1.0, steps=4 * FEATURE_DIM).reshape(4, FEATURE_DIM)
        self.transfer_max_q_error = float(
            (self.reference_residual_net(probe) - self.residual_net(probe))
            .abs()
            .max()
            .item()
        )


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_metadata(self):
    return checkpoint_metadata(
        self.source_model_sha256,
        c1_source_sha256=self.c1_source_sha256,
    )


def _atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _evaluation_payload(self):
    return {
        "metadata": _checkpoint_metadata(self),
        "residual_state_dict": self.residual_net.state_dict(),
        "decision_step": self.decision_step,
        "source_transition_count": self.source_transition_count,
        "transition_count": self.transition_count,
        "optimizer_step": self.optimizer_step,
        "transfer_max_q_error": self.transfer_max_q_error,
    }


def _save_evaluation_model(self):
    _atomic_torch_save(_evaluation_payload(self), MODEL_PATH)


def _save_model_snapshot(self):
    name = "m2_4_{}_t{:07d}.pt".format(BRANCH, self.transition_count)
    path = MODEL_SNAPSHOT_DIR / name
    if path.exists():
        raise FileExistsError("Refusing to overwrite model snapshot: {}".format(path))
    _atomic_torch_save(_evaluation_payload(self), path)


def _save_training_checkpoint(self):
    if self.pending_transition is not None or len(self.n_step_accumulator):
        raise RuntimeError("M2-4 checkpoint requested before the round boundary")
    if self.teacher_records:
        raise RuntimeError("M2-4 checkpoint requested with stale teacher records")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": _checkpoint_metadata(self),
        "residual_state_dict": self.residual_net.state_dict(),
        "target_residual_state_dict": self.target_residual_net.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "replay_buffer": self.replay_buffer.state_dict(),
        "teacher_buffer": self.teacher_buffer.state_dict(),
        "teacher_state": self.rule_teacher.state_dict() if self.rule_teacher else None,
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
        "teacher_window": dict(self.teacher_window),
        "planner_requests": self.temporal_planner.requests,
        "planner_computations": self.temporal_planner.computations,
        "planner_cache_hits": self.temporal_planner.cache_hits,
        "numpy_rng_state": self.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    _atomic_torch_save(payload, TRAINING_CHECKPOINT_PATH)


def _load_training_checkpoint(self):
    if not TRAINING_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError("No M2-4 training checkpoint is available to resume")
    try:
        checkpoint = torch.load(
            TRAINING_CHECKPOINT_PATH, map_location=self.device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(TRAINING_CHECKPOINT_PATH, map_location=self.device)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("config_hash") != config_hash():
        raise ValueError("M2-4 checkpoint configuration does not match this branch")
    if metadata.get("source_model_sha256") != self.source_model_sha256:
        raise ValueError("The frozen formal M2-3 source model changed")
    if BRANCH == "c3" and metadata.get("c1_source_sha256") != self.c1_source_sha256:
        raise ValueError("The selected C1 source model changed")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported M2-4 checkpoint format")

    self.residual_net.load_state_dict(checkpoint["residual_state_dict"], strict=True)
    self.target_residual_net.load_state_dict(
        checkpoint["target_residual_state_dict"], strict=True
    )
    self.target_residual_net.eval()
    for parameter in self.target_residual_net.parameters():
        parameter.requires_grad_(False)
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    self.teacher_buffer.load_state_dict(checkpoint["teacher_buffer"])
    if self.rule_teacher is not None:
        if checkpoint["teacher_state"] is None:
            raise ValueError("C2 checkpoint is missing the teacher state")
        self.rule_teacher.load_state_dict(checkpoint["teacher_state"])
    self.n_step_accumulator.load_state_dict(checkpoint["n_step_accumulator"])
    if len(self.n_step_accumulator):
        raise ValueError("A round-boundary checkpoint has a non-empty n-step queue")

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
    self.teacher_window = dict(checkpoint["teacher_window"])
    self.temporal_planner.requests = int(checkpoint["planner_requests"])
    self.temporal_planner.computations = int(checkpoint["planner_computations"])
    self.temporal_planner.cache_hits = int(checkpoint["planner_cache_hits"])
    self.rng.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"])
    self.pending_transition = None
    self.teacher_records = {}
