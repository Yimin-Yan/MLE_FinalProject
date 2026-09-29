"""Training callbacks for frozen-C1 residual rehearsal learning."""

from dataclasses import dataclass
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import events as e

from ...m2_3.features import potential_from_features, state_to_features
from ...m2_3.n_step import NStepAccumulator, OneStepTransition
from ...m2_3.replay_buffer import LAPReplayBuffer
from .callbacks import _fresh_shield_window
from .config import (
    ACTIONS,
    ACTION_TO_INDEX,
    ADAM_BETAS,
    ADAM_EPS,
    BASE_FEATURE_DIM,
    CHECKPOINT_AFTER_SESSION_ROUNDS,
    CHECKPOINT_FORMAT_VERSION,
    C1_TRAINING_CHECKPOINT_PATH,
    DISTILL_LAMBDA_MAX,
    DISTILL_LAMBDA_MIN,
    DISTILL_TARGET_RATIO,
    DISTILL_TEMPERATURE,
    FAIL_ON_STATIC_INVALID,
    FEATURE_DIM,
    FULL_CHECKPOINT_INTERVAL,
    GAMMA,
    GRAD_CLIP_NORM,
    LAP_ALPHA,
    LAP_PRIORITY_FLOOR,
    LEAGUE_BATCH_SIZE,
    LEAGUE_TD_WEIGHT,
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
    REHEARSAL_BATCH_SIZE,
    REHEARSAL_TD_WEIGHT,
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
    epsilon_for_transition,
)
from .model import C3ResidualAdapter, mask_q_values
from .rehearsal_buffer import FrozenRehearsalBuffer


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
        "loss_sum": 0.0,
        "td_loss_sum": 0.0,
        "league_loss_sum": 0.0,
        "rehearsal_loss_sum": 0.0,
        "distill_loss_sum": 0.0,
        "distill_lambda_sum": 0.0,
        "distill_ratio_sum": 0.0,
        "td_abs_sum": 0.0,
        "td_abs_max": 0.0,
        "grad_norm_sum": 0.0,
        "q_sum": 0.0,
        "q_count": 0,
        "q_abs_max": 0.0,
        "frozen_square_sum": 0.0,
        "adapter_square_sum": 0.0,
        "value_count": 0,
        "sample_priority_sum": 0.0,
        "temporal_min": float("inf"),
        "temporal_max": float("-inf"),
    }


def _load_file(path, map_location, weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rehearsal_buffer(self):
    if not C1_TRAINING_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "C3-v2 needs the C1 training checkpoint at {}".format(
                C1_TRAINING_CHECKPOINT_PATH
            )
        )
    checkpoint = _load_file(
        C1_TRAINING_CHECKPOINT_PATH, self.device, weights_only=False
    )
    metadata = checkpoint.get("metadata", {})
    if metadata.get("variant_id") != "M2-4-C1":
        raise ValueError("The rehearsal checkpoint does not belong to C1")
    if int(checkpoint.get("transition_count", -1)) != 250_000:
        raise ValueError("C1 rehearsal must come from the completed 250k run")
    if metadata.get("source_model_sha256") != self.m2_3_source_sha256:
        raise ValueError("C1 rehearsal and the frozen C1 model use different M2-3 sources")
    self.rehearsal_buffer = FrozenRehearsalBuffer(
        checkpoint["replay_buffer"],
        FEATURE_DIM,
        LAP_ALPHA,
        LAP_PRIORITY_FLOOR,
    )
    if len(self.rehearsal_buffer) != REPLAY_CAPACITY:
        raise ValueError("The frozen C1 rehearsal buffer must contain 100000 items")
    self.rehearsal_source_sha256 = _sha256(C1_TRAINING_CHECKPOINT_PATH)


def setup_training(self):
    """Create fresh C3 optimization state around the frozen C1 policy."""
    self.target_adapter_net = C3ResidualAdapter(FEATURE_DIM, len(ACTIONS)).to(self.device)
    self.optimizer = torch.optim.Adam(
        self.adapter_net.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
    )
    self.replay_buffer = LAPReplayBuffer(
        REPLAY_CAPACITY, FEATURE_DIM, LAP_ALPHA, LAP_PRIORITY_FLOOR
    )
    _load_rehearsal_buffer(self)
    self.n_step_accumulator = NStepAccumulator(N_STEP, GAMMA)
    self.pending_transition = None
    self.source_transition_count = 0
    self.transition_count = 0
    self.optimizer_step = 0
    self.rounds_completed = 0
    self.session_rounds = 0
    self.next_checkpoint_at = FULL_CHECKPOINT_INTERVAL
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

    MODEL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RESUME_TRAINING:
        _load_training_checkpoint(self)
        self.logger.info(
            "Resumed C3-v2 at transition %d", self.transition_count
        )
        return

    old_snapshots = sorted(MODEL_SNAPSHOT_DIR.glob("m2_4_c3_t*.pt"))
    if old_snapshots or TRAINING_CHECKPOINT_PATH.is_file():
        raise FileExistsError("A fresh C3-v2 run cannot overwrite existing checkpoints")
    with open(METRICS_PATH, "w", encoding="utf-8"):
        pass
    if self.transfer_max_q_error > TRANSFER_TOLERANCE:
        raise RuntimeError("The zero adapter changed C1 Q-values at t0")

    self.target_adapter_net.load_state_dict(self.adapter_net.state_dict(), strict=True)
    self.target_adapter_net.eval()
    for parameter in self.target_adapter_net.parameters():
        parameter.requires_grad_(False)
    _write_metric(
        {
            "kind": "transfer",
            "variant_id": VARIANT_ID,
            "source_branch": "c1",
            "m2_3_source_sha256": self.m2_3_source_sha256,
            "c1_source_sha256": self.c1_source_sha256,
            "rehearsal_source_sha256": self.rehearsal_source_sha256,
            "max_q_error": self.transfer_max_q_error,
            "optimizer": "new_adam",
            "league_replay": "fresh_lap",
            "rehearsal_replay": "frozen_c1_100000",
        }
    )
    _save_model_snapshot(self)
    _save_evaluation_model(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if self.pending_transition is not None:
        _commit_transition(self, self.pending_transition, done=False)
    if old_game_state is None or new_game_state is None:
        raise ValueError("A surviving step must provide old and new game states")
    if self_action not in ACTION_TO_INDEX:
        raise ValueError("Unknown action returned by agent: " + repr(self_action))
    self.pending_transition = _make_pending(
        self, old_game_state, self_action, new_game_state, events
    )


def end_of_round(self, last_game_state, last_action, events):
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
    if len(self.n_step_accumulator):
        raise RuntimeError("The n-step queue was not empty at the end of a round")
    if self.source_transition_count != self.transition_count:
        raise RuntimeError("Source and stored transition counts diverged")
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

    if self.transition_count >= SESSION_TRANSITION_LIMIT and self.metric_window["updates"]:
        _flush_training_metrics(self)
    if self.transition_count >= self.next_checkpoint_at:
        while self.next_checkpoint_at <= self.transition_count:
            self.next_checkpoint_at += FULL_CHECKPOINT_INTERVAL
        _save_training_checkpoint(self)
        _save_evaluation_model(self)
    elif CHECKPOINT_AFTER_SESSION_ROUNDS and self.session_rounds >= CHECKPOINT_AFTER_SESSION_ROUNDS:
        _save_training_checkpoint(self)
        _save_evaluation_model(self)


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
        round_number=round_number,
        step_number=step_number,
        events=tuple(events),
    )


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
            raise RuntimeError("The final n-step flush changed the C3-v2 budget")
        self.logger.info("Reached C3-v2 target %d", self.transition_count)


def _store_and_learn(self, transition):
    if self.transition_count >= MAX_TRAINING_TRANSITIONS:
        raise RuntimeError("Tried to store a transition beyond the 500k budget")
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
        available = bool(transition.old_mask[transition.action_index])
        if available:
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


def _tensor_batch(batch, device):
    return {
        "states": torch.from_numpy(batch["states"]).to(device),
        "actions": torch.from_numpy(batch["actions"]).to(device),
        "rewards": torch.from_numpy(batch["rewards"]).to(device),
        "next_states": torch.from_numpy(batch["next_states"]).to(device),
        "dones": torch.from_numpy(batch["dones"]).to(device),
        "next_masks": torch.from_numpy(batch["next_masks"]).to(device),
        "horizons": torch.from_numpy(batch["horizons"]).to(device),
    }


def _frozen_values(self, states):
    return self.base_net(states) + self.c1_residual_net(states)


def _td_values(self, data):
    states = data["states"]
    frozen = _frozen_values(self, states)
    adapter = self.adapter_net(states)
    total = frozen + adapter
    chosen = total.gather(1, data["actions"].unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_values = torch.zeros_like(data["rewards"])
        live = ~data["dones"]
        if live.any():
            next_states = data["next_states"][live]
            frozen_next = _frozen_values(self, next_states)
            online_next = frozen_next + self.adapter_net(next_states)
            next_actions = mask_q_values(
                online_next, data["next_masks"][live]
            ).argmax(dim=1)
            target_next = frozen_next + self.target_adapter_net(next_states)
            next_values[live] = target_next.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
        discounts = torch.pow(
            torch.full_like(data["rewards"], GAMMA),
            data["horizons"].to(dtype=data["rewards"].dtype),
        )
        targets = data["rewards"] + discounts * next_values
    return chosen, targets, frozen, adapter


def _learn_once(self):
    league_raw = self.replay_buffer.sample(LEAGUE_BATCH_SIZE, self.rng)
    rehearsal_raw = self.rehearsal_buffer.sample(REHEARSAL_BATCH_SIZE, self.rng)
    league = _tensor_batch(league_raw, self.device)
    rehearsal = _tensor_batch(rehearsal_raw, self.device)

    self.adapter_net.train()
    league_values, league_targets, frozen_values, adapter_values = _td_values(
        self, league
    )
    rehearsal_values, rehearsal_targets, frozen_rehearsal, _ = _td_values(
        self, rehearsal
    )
    league_loss = F.smooth_l1_loss(league_values, league_targets, beta=1.0)
    rehearsal_loss = F.smooth_l1_loss(
        rehearsal_values, rehearsal_targets, beta=1.0
    )
    td_loss = (
        LEAGUE_TD_WEIGHT * league_loss
        + REHEARSAL_TD_WEIGHT * rehearsal_loss
    )

    temperature = DISTILL_TEMPERATURE
    with torch.no_grad():
        teacher_probabilities = torch.softmax(frozen_rehearsal / temperature, dim=1)
    student_log_probabilities = torch.log_softmax(
        (frozen_rehearsal + self.adapter_net(rehearsal["states"])) / temperature,
        dim=1,
    )
    distill_loss = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    ) * (temperature ** 2)
    with torch.no_grad():
        distill_lambda = torch.clamp(
            DISTILL_TARGET_RATIO * td_loss.detach() / (distill_loss.detach() + 1e-8),
            min=DISTILL_LAMBDA_MIN,
            max=DISTILL_LAMBDA_MAX,
        )
    total_loss = td_loss + distill_lambda * distill_loss
    if not all(
        torch.isfinite(value)
        for value in (league_loss, rehearsal_loss, distill_loss, total_loss)
    ):
        raise FloatingPointError("Non-finite C3-v2 loss")

    self.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.adapter_net.parameters(), GRAD_CLIP_NORM
    )
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Non-finite C3-v2 gradient norm")
    self.optimizer.step()
    self.optimizer_step += 1

    league_td = league_targets - league_values
    self.replay_buffer.update_from_td_errors(
        league_raw["indices"], league_td.detach().cpu().numpy()
    )
    if self.optimizer_step % TARGET_UPDATE_INTERVAL == 0:
        self.target_adapter_net.load_state_dict(
            self.adapter_net.state_dict(), strict=True
        )
    _record_update(
        self,
        total_loss,
        td_loss,
        league_loss,
        rehearsal_loss,
        distill_loss,
        distill_lambda,
        league_td,
        grad_norm,
        league_values,
        frozen_values,
        adapter_values,
        league_raw,
    )
    if self.optimizer_step % METRIC_LOG_INTERVAL == 0:
        _flush_training_metrics(self)


def _record_update(
    self,
    total_loss,
    td_loss,
    league_loss,
    rehearsal_loss,
    distill_loss,
    distill_lambda,
    league_td,
    grad_norm,
    chosen_values,
    frozen_values,
    adapter_values,
    league_raw,
):
    window = self.metric_window
    window["updates"] += 1
    window["loss_sum"] += float(total_loss.item())
    window["td_loss_sum"] += float(td_loss.item())
    window["league_loss_sum"] += float(league_loss.item())
    window["rehearsal_loss_sum"] += float(rehearsal_loss.item())
    window["distill_loss_sum"] += float(distill_loss.item())
    window["distill_lambda_sum"] += float(distill_lambda.item())
    contribution = float((distill_lambda * distill_loss).item())
    window["distill_ratio_sum"] += contribution / max(float(td_loss.item()), 1e-8)
    window["td_abs_sum"] += float(league_td.detach().abs().mean().item())
    window["td_abs_max"] = max(
        window["td_abs_max"], float(league_td.detach().abs().max().item())
    )
    window["grad_norm_sum"] += float(grad_norm.item())
    window["q_sum"] += float(chosen_values.detach().sum().item())
    window["q_count"] += int(chosen_values.numel())
    window["q_abs_max"] = max(
        window["q_abs_max"], float(chosen_values.detach().abs().max().item())
    )
    window["frozen_square_sum"] += float(frozen_values.detach().square().sum().item())
    window["adapter_square_sum"] += float(adapter_values.detach().square().sum().item())
    window["value_count"] += int(frozen_values.numel())
    window["sample_priority_sum"] += float(league_raw["raw_priorities"].mean())


def _flush_training_metrics(self):
    window = self.metric_window
    updates = max(window["updates"], 1)
    q_count = max(window["q_count"], 1)
    value_count = max(window["value_count"], 1)
    priority_mean, priority_max, above_floor = self.replay_buffer.priority_statistics()
    frozen_rms = (window["frozen_square_sum"] / value_count) ** 0.5
    adapter_rms = (window["adapter_square_sum"] / value_count) ** 0.5
    temporal_min = window["temporal_min"]
    temporal_max = window["temporal_max"]
    if not np.isfinite(temporal_min):
        temporal_min = 0.0
    if not np.isfinite(temporal_max):
        temporal_max = 0.0
    shield = self.shield_window
    legal_total = max(shield["legal_actions"], 1)
    bomb_total = max(shield["bomb_candidates"], 1)
    _write_metric(
        {
            "kind": "training",
            "source_transition_count": self.source_transition_count,
            "transition_count": self.transition_count,
            "decision_step": self.decision_step,
            "optimizer_step": self.optimizer_step,
            "curriculum_block": os.environ.get(
                "SEAT_M2_4_CURRICULUM_BLOCK", "unspecified"
            ),
            "epsilon": epsilon_for_transition(self.source_transition_count),
            "learning_rate": LEARNING_RATE,
            "loss_mean": window["loss_sum"] / updates,
            "td_loss_mean": window["td_loss_sum"] / updates,
            "league_td_loss_mean": window["league_loss_sum"] / updates,
            "rehearsal_td_loss_mean": window["rehearsal_loss_sum"] / updates,
            "distill_loss_mean": window["distill_loss_sum"] / updates,
            "distill_lambda_mean": window["distill_lambda_sum"] / updates,
            "distill_to_td_ratio": window["distill_ratio_sum"] / updates,
            "td_abs_mean": window["td_abs_sum"] / updates,
            "td_abs_max": window["td_abs_max"],
            "grad_norm_mean": window["grad_norm_sum"] / updates,
            "q_mean": window["q_sum"] / q_count,
            "q_abs_max": window["q_abs_max"],
            "frozen_c1_q_rms": frozen_rms,
            "adapter_q_rms": adapter_rms,
            "adapter_to_c1_rms": adapter_rms / max(frozen_rms, 1e-8),
            "league_batch_size": LEAGUE_BATCH_SIZE,
            "rehearsal_batch_size": REHEARSAL_BATCH_SIZE,
            "league_buffer_size": len(self.replay_buffer),
            "rehearsal_buffer_size": len(self.rehearsal_buffer),
            "sample_priority_mean": window["sample_priority_sum"] / updates,
            "replay_priority_mean": priority_mean,
            "replay_priority_max": priority_max,
            "priority_above_floor_fraction": above_floor,
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
    )
    self.metric_window = _fresh_metric_window()
    self.shield_window = _fresh_shield_window()


def _metadata(self):
    return checkpoint_metadata(self.m2_3_source_sha256, self.c1_source_sha256)


def _atomic_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _evaluation_payload(self):
    return {
        "metadata": _metadata(self),
        "adapter_state_dict": self.adapter_net.state_dict(),
        "decision_step": self.decision_step,
        "source_transition_count": self.source_transition_count,
        "transition_count": self.transition_count,
        "optimizer_step": self.optimizer_step,
        "transfer_max_q_error": self.transfer_max_q_error,
    }


def _save_evaluation_model(self):
    _atomic_save(_evaluation_payload(self), MODEL_PATH)


def _save_model_snapshot(self):
    path = MODEL_SNAPSHOT_DIR / "m2_4_c3_t{:07d}.pt".format(
        self.transition_count
    )
    if path.exists():
        raise FileExistsError("Refusing to overwrite C3-v2 snapshot: {}".format(path))
    _atomic_save(_evaluation_payload(self), path)


def _save_training_checkpoint(self):
    if self.pending_transition is not None or len(self.n_step_accumulator):
        raise RuntimeError("C3-v2 checkpoint requested before a round boundary")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": _metadata(self),
        "adapter_state_dict": self.adapter_net.state_dict(),
        "target_adapter_state_dict": self.target_adapter_net.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "replay_buffer": self.replay_buffer.state_dict(),
        "n_step_accumulator": self.n_step_accumulator.state_dict(),
        "decision_step": self.decision_step,
        "source_transition_count": self.source_transition_count,
        "transition_count": self.transition_count,
        "optimizer_step": self.optimizer_step,
        "rounds_completed": self.rounds_completed,
        "next_checkpoint_at": self.next_checkpoint_at,
        "static_invalid_count": self.static_invalid_count,
        "race_invalid_count": self.race_invalid_count,
        "action_counts": dict(self.action_counts),
        "reward_totals": dict(self.reward_totals),
        "metric_window": dict(self.metric_window),
        "shield_window": dict(self.shield_window),
        "planner_requests": self.temporal_planner.requests,
        "planner_computations": self.temporal_planner.computations,
        "planner_cache_hits": self.temporal_planner.cache_hits,
        "numpy_rng_state": self.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "rehearsal_source_sha256": self.rehearsal_source_sha256,
    }
    _atomic_save(payload, TRAINING_CHECKPOINT_PATH)


def _load_training_checkpoint(self):
    if not TRAINING_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError("No C3-v2 training checkpoint is available")
    checkpoint = _load_file(TRAINING_CHECKPOINT_PATH, self.device, weights_only=False)
    if int(checkpoint.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported C3-v2 training checkpoint format")
    metadata = checkpoint.get("metadata", {})
    if metadata.get("config_hash") != config_hash():
        raise ValueError("C3-v2 checkpoint configuration changed")
    if metadata.get("m2_3_source_sha256") != self.m2_3_source_sha256:
        raise ValueError("The frozen M2-3 source changed")
    if metadata.get("c1_source_sha256") != self.c1_source_sha256:
        raise ValueError("The frozen C1 source changed")
    if checkpoint.get("rehearsal_source_sha256") != self.rehearsal_source_sha256:
        raise ValueError("The frozen C1 rehearsal source changed")
    self.adapter_net.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    self.target_adapter_net.load_state_dict(
        checkpoint["target_adapter_state_dict"], strict=True
    )
    self.target_adapter_net.eval()
    for parameter in self.target_adapter_net.parameters():
        parameter.requires_grad_(False)
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    self.n_step_accumulator.load_state_dict(checkpoint["n_step_accumulator"])
    if len(self.n_step_accumulator):
        raise ValueError("A round-boundary checkpoint has a non-empty n-step queue")
    self.decision_step = int(checkpoint["decision_step"])
    self.source_transition_count = int(checkpoint["source_transition_count"])
    self.transition_count = int(checkpoint["transition_count"])
    self.optimizer_step = int(checkpoint["optimizer_step"])
    self.rounds_completed = int(checkpoint["rounds_completed"])
    self.next_checkpoint_at = int(checkpoint["next_checkpoint_at"])
    self.static_invalid_count = int(checkpoint["static_invalid_count"])
    self.race_invalid_count = int(checkpoint["race_invalid_count"])
    self.action_counts = dict(checkpoint["action_counts"])
    self.reward_totals = dict(checkpoint["reward_totals"])
    self.metric_window = dict(checkpoint["metric_window"])
    self.shield_window = dict(checkpoint["shield_window"])
    self.temporal_planner.requests = int(checkpoint["planner_requests"])
    self.temporal_planner.computations = int(checkpoint["planner_computations"])
    self.temporal_planner.cache_hits = int(checkpoint["planner_cache_hits"])
    self.rng.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"])
    self.pending_transition = None


def _write_metric(data):
    with open(METRICS_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, sort_keys=True) + "\n")
