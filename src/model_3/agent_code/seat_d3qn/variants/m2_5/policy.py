"""Pure action-selection helpers for the recovery gate."""

import numpy as np


BOMB_INDEX = 5


def legal_mask_from_features(features):
    legal = np.zeros(6, dtype=np.bool_)
    legal[:4] = np.asarray(features[:4]) > 0.5
    legal[4] = True
    legal[5] = bool(features[4] > 0.5)
    return legal


def greedy_legal_action(q_values, legal_mask):
    masked = np.where(legal_mask, q_values, -np.inf)
    return int(np.argmax(masked))


def select_recovery_action(
    q_values,
    risk_scores,
    legal_mask,
    known_hazard,
    risk_threshold,
    risk_margin,
    q_gap_cap,
):
    """Return the source action unless a low-cost recovery move is available."""
    q_values = np.asarray(q_values, dtype=np.float64)
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=np.bool_)
    base = greedy_legal_action(q_values, legal_mask)
    if not known_hazard or base == BOMB_INDEX:
        return base, False
    if risk_scores[base] < risk_threshold:
        return base, False

    candidates = legal_mask.copy()
    candidates[BOMB_INDEX] = False
    candidates &= risk_scores <= risk_scores[base] - risk_margin
    candidates &= q_values >= q_values[base] - q_gap_cap
    if not np.any(candidates):
        return base, False
    chosen = greedy_legal_action(q_values, candidates)
    return chosen, chosen != base

