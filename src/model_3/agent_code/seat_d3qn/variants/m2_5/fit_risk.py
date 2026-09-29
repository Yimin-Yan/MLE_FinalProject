"""Fit the M2-5 risk model after all ten collection blocks are complete."""

from copy import deepcopy
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    ACTIONS,
    BATCH_SIZE,
    BLOCK_TRANSITIONS,
    COLLECTION_BLOCKS,
    DATASET_DIR,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MAX_COLLECTION_TRANSITIONS,
    MAX_EPOCHS,
    MAX_INTERVENTION_RATE,
    MIN_AVERAGE_PRECISION,
    MIN_INTERVENTION_RATE,
    MIN_POSITIVES_PER_DOMAIN,
    MIN_POSITIVES_PER_RECOVERY_ACTION,
    MIN_ROC_AUC,
    MIN_TOTAL_POSITIVES,
    POS_WEIGHT_CAP,
    Q_GAP_CAP,
    RISK_CHECKPOINT_PATH,
    RISK_MARGIN,
    RISK_THRESHOLD_CANDIDATES,
    TRAINING_SUMMARY_PATH,
    WEIGHT_DECAY,
    checkpoint_metadata,
    config_hash,
    MASTER_SEED,
)
from .dataset import FIELDS, load_shard
from .model import WastefulDeathRisk, chosen_action_logits
from .source import FrozenC3Policy


def shard_path(block_index, domain):
    return DATASET_DIR / "block_{:02d}_{}.npz".format(block_index, domain)


def load_dataset():
    chunks = {name: [] for name in FIELDS}
    block_counts = []
    for index, (domain, _) in enumerate(COLLECTION_BLOCKS):
        path = shard_path(index, domain)
        data = load_shard(path)
        count = len(data["features"])
        if count != BLOCK_TRANSITIONS:
            raise ValueError("Collection block {} has {} rows".format(index, count))
        block_counts.append(count)
        for name in FIELDS:
            chunks[name].append(data[name])
    merged = {name: np.concatenate(chunks[name], axis=0) for name in FIELDS}
    if len(merged["features"]) != MAX_COLLECTION_TRANSITIONS:
        raise ValueError("M2-5 did not collect exactly 250000 transitions")
    return merged, block_counts


def _round_bucket(round_uid):
    digest = hashlib.blake2b(str(int(round_uid)).encode("ascii"), digest_size=2).digest()
    return int.from_bytes(digest, "little") % 20


def split_by_round(round_ids):
    split = np.empty(len(round_ids), dtype=np.int8)
    for round_uid in np.unique(round_ids):
        bucket = _round_bucket(round_uid)
        if bucket < 14:
            value = 0
        elif bucket < 17:
            value = 1
        else:
            value = 2
        split[round_ids == round_uid] = value
    return split


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positives)


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def classification_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    return {
        "count": int(len(labels)),
        "positives": int(labels.sum()),
        "positive_fraction": float(labels.mean()) if len(labels) else 0.0,
        "roc_auc": float(roc_auc(labels, scores)),
        "average_precision": float(average_precision(labels, scores)),
    }


def _predict_chosen(model, features, actions):
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 8192):
            states = torch.from_numpy(features[start : start + 8192])
            chosen = torch.from_numpy(actions[start : start + 8192].astype(np.int64))
            logits = chosen_action_logits(model, states, chosen)
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


def _predict_all(model, features):
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 8192):
            states = torch.from_numpy(features[start : start + 8192])
            output.append(torch.sigmoid(model(states)).cpu().numpy())
    return np.concatenate(output, axis=0)


def _source_q(features):
    source = FrozenC3Policy("cpu")
    output = []
    with torch.inference_mode():
        for start in range(0, len(features), 8192):
            states = torch.from_numpy(features[start : start + 8192])
            output.append(source(states).cpu().numpy())
    return np.concatenate(output, axis=0)


def _legal_masks(features):
    legal = np.zeros((len(features), len(ACTIONS)), dtype=np.bool_)
    legal[:, :4] = features[:, :4] > 0.5
    legal[:, 4] = True
    legal[:, 5] = features[:, 4] > 0.5
    return legal


def gate_statistics(features, actions, labels, explored, q_values, risks, threshold):
    legal = _legal_masks(features)
    masked_q = np.where(legal, q_values, -np.inf)
    base = masked_q.argmax(axis=1)
    rows = np.arange(len(features))
    base_q = q_values[rows, base]
    base_risk = risks[rows, base]

    candidates = legal.copy()
    candidates[:, 5] = False
    candidates &= risks <= base_risk[:, None] - RISK_MARGIN
    candidates &= q_values >= base_q[:, None] - Q_GAP_CAP
    recovery = features[:, 60] > 0.5
    recovery &= base != 5
    recovery &= base_risk >= threshold
    changed = recovery & candidates.any(axis=1)

    on_policy = (~explored) & (actions == base)
    positive = on_policy & (labels == 1)
    caught = positive & changed
    return {
        "threshold": float(threshold),
        "interventions": int(changed.sum()),
        "intervention_rate": float(changed.mean()),
        "on_policy_examples": int(on_policy.sum()),
        "on_policy_positives": int(positive.sum()),
        "positive_interventions": int(caught.sum()),
        "positive_coverage": float(caught.sum() / max(1, positive.sum())),
        "bomb_source_overrides": int((changed & (base == 5)).sum()),
        "bomb_candidates": 0,
    }


def _train_model(features, actions, labels, split):
    train_idx = np.flatnonzero(split == 0)
    validation_idx = np.flatnonzero(split == 1)
    positives = int(labels[train_idx].sum())
    negatives = len(train_idx) - positives
    if positives == 0:
        raise RuntimeError("The M2-5 training split has no positive labels")
    positive_weight = min(negatives / positives, POS_WEIGHT_CAP)

    torch.manual_seed(MASTER_SEED)
    rng = np.random.default_rng(MASTER_SEED)
    model = WastefulDeathRisk()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_state = None
    best_ap = -1.0
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        shuffled = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(shuffled), BATCH_SIZE):
            indices = shuffled[start : start + BATCH_SIZE]
            states = torch.from_numpy(features[indices])
            chosen = torch.from_numpy(actions[indices].astype(np.int64))
            targets = torch.from_numpy(labels[indices].astype(np.float32))
            logits = chosen_action_logits(model, states, chosen)
            weights = torch.where(
                targets > 0.5,
                torch.full_like(targets, positive_weight),
                torch.ones_like(targets),
            )
            loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            losses.append(float(loss.item()))

        validation_scores = _predict_chosen(
            model, features[validation_idx], actions[validation_idx]
        )
        metrics = classification_metrics(labels[validation_idx], validation_scores)
        metrics["epoch"] = epoch
        metrics["training_loss"] = float(np.mean(losses))
        history.append(metrics)
        if metrics["average_precision"] > best_ap + 1e-8:
            best_ap = metrics["average_precision"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("M2-5 risk fitting did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, positive_weight, best_epoch, history


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def main():
    data, block_counts = load_dataset()
    labels = data["labels"]
    if int(labels.sum()) < MIN_TOTAL_POSITIVES:
        raise RuntimeError(
            "M2-5 collected only {} positive labels; at least {} are required".format(
                int(labels.sum()), MIN_TOTAL_POSITIVES
            )
        )
    action_positives = {
        ACTIONS[index]: int(((data["actions"] == index) & (labels == 1)).sum())
        for index in range(len(ACTIONS))
    }
    for action in ACTIONS[:5]:
        if action_positives[action] < MIN_POSITIVES_PER_RECOVERY_ACTION:
            raise RuntimeError(
                "M2-5 has only {} positive labels for {}".format(
                    action_positives[action], action
                )
            )
    domain_positives = {
        str(domain_id): int(((data["domain_ids"] == domain_id) & (labels == 1)).sum())
        for domain_id in range(3)
    }
    for domain_id, count in domain_positives.items():
        if count < MIN_POSITIVES_PER_DOMAIN:
            raise RuntimeError(
                "M2-5 has only {} positive labels in domain {}".format(
                    count, domain_id
                )
            )
    split = split_by_round(data["round_ids"])
    for split_id, name in enumerate(("train", "validation", "test")):
        if not np.any(split == split_id):
            raise RuntimeError("The {} split is empty".format(name))
        for domain_id in range(3):
            if not np.any((split == split_id) & (data["domain_ids"] == domain_id)):
                raise RuntimeError("The {} split is missing domain {}".format(name, domain_id))

    model, positive_weight, best_epoch, history = _train_model(
        data["features"], data["actions"], labels, split
    )
    chosen_scores = _predict_chosen(model, data["features"], data["actions"])
    all_risks = _predict_all(model, data["features"])
    q_values = _source_q(data["features"])

    split_metrics = {}
    for split_id, name in enumerate(("train", "validation", "test")):
        mask = split == split_id
        split_metrics[name] = classification_metrics(labels[mask], chosen_scores[mask])
    test_metrics = split_metrics["test"]
    print(
        "Held-out risk metrics: AUC={:.4f}, AP={:.4f}".format(
            test_metrics["roc_auc"], test_metrics["average_precision"]
        ),
        flush=True,
    )
    if test_metrics["roc_auc"] < MIN_ROC_AUC:
        raise RuntimeError(
            "M2-5 held-out ROC AUC {:.4f} is below {:.2f}".format(
                test_metrics["roc_auc"], MIN_ROC_AUC
            )
        )
    if test_metrics["average_precision"] < MIN_AVERAGE_PRECISION:
        raise RuntimeError(
            "M2-5 held-out average precision {:.4f} is below {:.2f}".format(
                test_metrics["average_precision"], MIN_AVERAGE_PRECISION
            )
        )

    validation_mask = split == 1
    candidates = []
    for threshold in RISK_THRESHOLD_CANDIDATES:
        stats = gate_statistics(
            data["features"][validation_mask],
            data["actions"][validation_mask],
            labels[validation_mask],
            data["explored"][validation_mask],
            q_values[validation_mask],
            all_risks[validation_mask],
            threshold,
        )
        candidates.append(stats)
    eligible = [
        item
        for item in candidates
        if MIN_INTERVENTION_RATE <= item["intervention_rate"] <= MAX_INTERVENTION_RATE
        and item["positive_interventions"] >= 2
        and item["bomb_source_overrides"] == 0
    ]
    if not eligible:
        raise RuntimeError("No conservative M2-5 risk threshold passed the offline gate")
    selected = max(
        eligible,
        key=lambda item: (
            item["positive_coverage"],
            item["positive_interventions"],
            item["threshold"],
        ),
    )
    test_mask = split == 2
    test_gate = gate_statistics(
        data["features"][test_mask],
        data["actions"][test_mask],
        labels[test_mask],
        data["explored"][test_mask],
        q_values[test_mask],
        all_risks[test_mask],
        selected["threshold"],
    )
    if not 0.00025 <= test_gate["intervention_rate"] <= 0.003:
        raise RuntimeError("M2-5 held-out gate rate is outside its conservative range")
    if test_gate["positive_interventions"] < 1:
        raise RuntimeError("M2-5 held-out gate caught no positive example")

    checkpoint = {
        "metadata": checkpoint_metadata(),
        "risk_state_dict": model.state_dict(),
        "risk_threshold": selected["threshold"],
        "risk_margin": RISK_MARGIN,
        "q_gap_cap": Q_GAP_CAP,
        "collection_transition_count": MAX_COLLECTION_TRANSITIONS,
        "positive_weight": float(positive_weight),
        "best_epoch": int(best_epoch),
        "split_metrics": split_metrics,
        "validation_gate": selected,
        "test_gate": test_gate,
    }
    RISK_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RISK_CHECKPOINT_PATH.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, RISK_CHECKPOINT_PATH)

    summary = {
        "algorithm_id": checkpoint_metadata()["algorithm_id"],
        "config_hash": config_hash(),
        "block_counts": block_counts,
        "total_transitions": int(len(labels)),
        "total_positives": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "positive_labels_by_action": action_positives,
        "positive_labels_by_domain": domain_positives,
        "exploration_fraction": float(data["explored"].mean()),
        "positive_weight": float(positive_weight),
        "best_epoch": int(best_epoch),
        "split_metrics": split_metrics,
        "threshold_candidates": candidates,
        "selected_validation_gate": selected,
        "held_out_test_gate": test_gate,
        "history": history,
        "risk_checkpoint": str(RISK_CHECKPOINT_PATH.resolve()),
    }
    _write_json(TRAINING_SUMMARY_PATH, summary)
    print("M2-5 collected exactly {} transitions".format(len(labels)))
    print("Wasteful-death positives: {} ({:.3%})".format(int(labels.sum()), labels.mean()))
    print(
        "Held-out risk metrics: AUC={:.4f}, AP={:.4f}".format(
            test_metrics["roc_auc"], test_metrics["average_precision"]
        )
    )
    print(
        "Selected risk threshold {:.3f}, offline intervention rate {:.3%}".format(
            selected["threshold"], selected["intervention_rate"]
        )
    )
    print("Risk checkpoint: {}".format(RISK_CHECKPOINT_PATH))


if __name__ == "__main__":
    main()
