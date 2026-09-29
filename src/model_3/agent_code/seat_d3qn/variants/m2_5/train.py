"""Collect action-labelled wasteful-death examples; task Q stays frozen."""

import os
from pathlib import Path

import numpy as np

from ..m2_3.features import state_to_features
from .config import (
    ACTION_TO_INDEX,
    BLOCK_INDEX_ENV,
    BLOCK_TRANSITIONS,
    COLLECTION_SEED_ENV,
    DOMAIN_ID_ENV,
    DOMAIN_IDS,
    SHARD_ENV,
)
from .dataset import append_rows, load_shard, save_shard
from .labels import wasteful_death_labels


def _state_key(game_state):
    if game_state is None:
        return None
    return int(game_state["round"]), int(game_state["step"])


def setup_training(self):
    raw_path = os.environ.get(SHARD_ENV)
    if not raw_path:
        raise ValueError("SEAT_M2_5_SHARD_PATH is required during collection")
    self.shard_path = Path(raw_path).expanduser().resolve()
    self.block_index = int(os.environ[BLOCK_INDEX_ENV])
    domain_name = os.environ[DOMAIN_ID_ENV]
    if domain_name not in DOMAIN_IDS:
        raise ValueError("Unknown M2-5 collection domain: " + domain_name)
    self.domain_id = DOMAIN_IDS[domain_name]
    self.collection_session_seed = int(os.environ[COLLECTION_SEED_ENV])
    self.shard = load_shard(self.shard_path)
    if len(self.shard["features"]) > BLOCK_TRANSITIONS:
        raise ValueError("M2-5 collection shard is larger than one block")
    self.round_records = []


def _record(self, game_state, action, event_list):
    if game_state is None or action not in ACTION_TO_INDEX:
        return
    key = _state_key(game_state)
    cached = self.last_observation
    if cached is not None and cached["key"] == key and cached["action"] == action:
        features = cached["features"].copy()
        explored = cached["explored"]
    else:
        analysis = self.temporal_planner.analyze(game_state, need_escape_labels=False)
        features = state_to_features(game_state, analysis)
        explored = False
    item = {
        "key": key,
        "features": features,
        "action": ACTION_TO_INDEX[action],
        "events": tuple(event_list),
        "explored": bool(explored),
        "step": int(game_state["step"]),
    }
    if self.round_records and self.round_records[-1]["key"] == key:
        self.round_records[-1] = item
    else:
        self.round_records.append(item)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _record(self, old_game_state, self_action, events)


def _round_rows(self):
    records = self.round_records
    labels, distances = wasteful_death_labels(records)
    count = len(records)
    if count == 0:
        return None
    round_number = int(records[0]["key"][0])
    round_uid = self.collection_session_seed * 1_000_000 + round_number
    return {
        "features": np.asarray([item["features"] for item in records], dtype=np.float32),
        "actions": np.asarray([item["action"] for item in records], dtype=np.int8),
        "labels": np.asarray(labels, dtype=np.uint8),
        "distances": np.asarray(distances, dtype=np.uint8),
        "explored": np.asarray([item["explored"] for item in records], dtype=np.bool_),
        "round_ids": np.full(count, round_uid, dtype=np.int64),
        "steps": np.asarray([item["step"] for item in records], dtype=np.int16),
        "domain_ids": np.full(count, self.domain_id, dtype=np.int8),
    }


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action, events)
    rows = _round_rows(self)
    if rows is not None:
        self.shard = append_rows(self.shard, rows, BLOCK_TRANSITIONS)
        save_shard(self.shard_path, self.shard)
    self.round_records = []
    self.collection_cooldown = 0
