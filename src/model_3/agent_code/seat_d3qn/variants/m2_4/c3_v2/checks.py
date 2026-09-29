"""Preflight checks shared by the C3-v2 scripts."""

import hashlib
import json

import torch

from .config import (
    C1_MODEL_PATH,
    C1_TRAINING_CHECKPOINT_PATH,
    GATE_TARGETS,
    MAX_TRAINING_TRANSITIONS,
    M2_3_MODEL_PATH,
    REHEARSAL_BATCH_SIZE,
    REPLAY_CAPACITY,
    REPOSITORY_ROOT,
    STAGE_NAMES,
    STAGE_OPPONENTS,
    STAGE_TARGETS,
)


C1_SELECTION_SUMMARY = (
    REPOSITORY_ROOT / "results/m2_4/c1/validation/selection-summary.json"
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch(path, weights_only=False):
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


def verify_c1_source():
    for path in (M2_3_MODEL_PATH, C1_MODEL_PATH, C1_TRAINING_CHECKPOINT_PATH):
        if not path.is_file():
            raise FileNotFoundError("Required C3-v2 source is missing: {}".format(path))

    m2_3_hash = sha256(M2_3_MODEL_PATH)
    c1_model = load_torch(C1_MODEL_PATH, weights_only=True)
    metadata = c1_model.get("metadata", {})
    if metadata.get("variant_id") != "M2-4-C1":
        raise ValueError("c1-model.pt is not an M2-4 C1 checkpoint")
    if int(c1_model.get("transition_count", -1)) != 250_000:
        raise ValueError("C3-v2 must start from selected C1 t0250000")
    if metadata.get("source_model_sha256") != m2_3_hash:
        raise ValueError("C1 was built from a different M2-3 checkpoint")

    training = load_torch(C1_TRAINING_CHECKPOINT_PATH, weights_only=False)
    training_metadata = training.get("metadata", {})
    if training_metadata.get("variant_id") != "M2-4-C1":
        raise ValueError("The C1 rehearsal source has the wrong variant")
    if int(training.get("transition_count", -1)) != 250_000:
        raise ValueError("The C1 rehearsal source is not the completed run")
    if training_metadata.get("source_model_sha256") != m2_3_hash:
        raise ValueError("C1 model and rehearsal use different M2-3 sources")
    replay = training.get("replay_buffer", {})
    if int(replay.get("size", -1)) != REPLAY_CAPACITY:
        raise ValueError("C3-v2 needs the full 100000-item C1 replay")
    if REHEARSAL_BATCH_SIZE >= int(replay["size"]):
        raise ValueError("The rehearsal batch is invalid")

    if C1_SELECTION_SUMMARY.is_file():
        with C1_SELECTION_SUMMARY.open("r", encoding="utf-8") as file:
            selection = json.load(file)
        selected = selection.get("best_snapshot", {})
        if int(selected.get("transition_count", -1)) != 250_000:
            raise ValueError("The C1 validation summary did not select t0250000")
    return {
        "m2_3_sha256": m2_3_hash,
        "c1_sha256": sha256(C1_MODEL_PATH),
        "rehearsal_sha256": sha256(C1_TRAINING_CHECKPOINT_PATH),
    }


def verify_schedule():
    expected_stages = tuple(
        range(50_000, MAX_TRAINING_TRANSITIONS + 1, 50_000)
    )
    expected_gates = tuple(
        range(25_000, MAX_TRAINING_TRANSITIONS + 1, 25_000)
    )
    if tuple(STAGE_TARGETS) != expected_stages:
        raise ValueError("C3-v2 must use ten 50k curriculum stages")
    if tuple(GATE_TARGETS) != expected_gates:
        raise ValueError("C3-v2 must checkpoint and gate every 25k")
    if len(STAGE_NAMES) != 10:
        raise ValueError("C3-v2 needs exactly ten named stages")

    expected_counts = {
        "official": 3,
        "rule_heavy": 2,
        "strong_cross": 2,
        "safety_resource": 2,
        "noisy_robustness": 1,
    }
    actual_counts = {name: STAGE_NAMES.count(name) for name in set(STAGE_NAMES)}
    if actual_counts != expected_counts:
        raise ValueError("The C3-v2 league proportions changed")
    for name in STAGE_NAMES:
        opponents = STAGE_OPPONENTS.get(name)
        if opponents is None or len(opponents) != 3:
            raise ValueError("Every C3-v2 stage needs three opponents")
        for opponent in opponents:
            if not (REPOSITORY_ROOT / "agent_code" / opponent).is_dir():
                raise FileNotFoundError("League opponent is missing: " + opponent)
    return expected_counts


def verify_all():
    sources = verify_c1_source()
    proportions = verify_schedule()
    return {"sources": sources, "stage_counts": proportions}
