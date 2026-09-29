"""Small utilities for the ten immutable collection shards."""

import os

import numpy as np

from .config import FEATURE_DIM


FIELDS = (
    "features",
    "actions",
    "labels",
    "distances",
    "explored",
    "round_ids",
    "steps",
    "domain_ids",
)


def empty_shard():
    return {
        "features": np.empty((0, FEATURE_DIM), dtype=np.float32),
        "actions": np.empty(0, dtype=np.int8),
        "labels": np.empty(0, dtype=np.uint8),
        "distances": np.empty(0, dtype=np.uint8),
        "explored": np.empty(0, dtype=np.bool_),
        "round_ids": np.empty(0, dtype=np.int64),
        "steps": np.empty(0, dtype=np.int16),
        "domain_ids": np.empty(0, dtype=np.int8),
    }


def _check(data):
    count = int(data["features"].shape[0])
    if data["features"].shape != (count, FEATURE_DIM):
        raise ValueError("M2-5 dataset has the wrong feature shape")
    for name in FIELDS[1:]:
        if data[name].shape != (count,):
            raise ValueError("M2-5 dataset field {} has the wrong length".format(name))
    return count


def load_shard(path):
    if not path.is_file():
        return empty_shard()
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(FIELDS):
            raise ValueError("M2-5 dataset shard has an unexpected schema")
        data = {name: saved[name] for name in FIELDS}
    _check(data)
    return data


def append_rows(data, rows, limit):
    count = _check(data)
    room = max(0, int(limit) - count)
    take = min(room, int(rows["features"].shape[0]))
    if take == 0:
        return data
    merged = {
        name: np.concatenate((data[name], rows[name][:take]), axis=0)
        for name in FIELDS
    }
    _check(merged)
    return merged


def save_shard(path, data):
    _check(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **data)
    os.replace(temporary, path)


def shard_count(path):
    if not path.is_file():
        return 0
    with np.load(path, allow_pickle=False) as saved:
        return int(saved["features"].shape[0])
