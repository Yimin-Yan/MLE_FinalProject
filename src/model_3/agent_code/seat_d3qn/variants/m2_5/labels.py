"""Build short-horizon labels from one completed round."""

import events as e

from .config import LABEL_HORIZON


def wasteful_death_labels(records, horizon=LABEL_HORIZON):
    labels = []
    distances = []
    for start in range(len(records)):
        stop = min(len(records), start + horizon)
        death_at = None
        useful_score = False
        for index in range(start, stop):
            observed = records[index]["events"]
            useful_score = useful_score or e.COIN_COLLECTED in observed
            useful_score = useful_score or e.KILLED_OPPONENT in observed
            if e.KILLED_SELF in observed:
                death_at = index
                break
        labels.append(int(death_at is not None and not useful_score))
        distances.append(0 if death_at is None else death_at - start + 1)
    return labels, distances

