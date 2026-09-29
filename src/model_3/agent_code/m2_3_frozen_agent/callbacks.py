"""Expose the trained M2-3 policy as an evaluation-only opponent."""

import os

from agent_code.seat_d3qn.variants.m2_3.callbacks import act
from agent_code.seat_d3qn.variants.m2_3.callbacks import setup as setup_m2_3
from agent_code.seat_d3qn.variants.m2_3.config import EVALUATION_MODEL_ENV


def setup(self):
    # Candidate evaluation paths belong to seat_d3qn, not this frozen opponent.
    candidate_path = os.environ.pop(EVALUATION_MODEL_ENV, None)
    try:
        setup_m2_3(self)
    finally:
        if candidate_path is not None:
            os.environ[EVALUATION_MODEL_ENV] = candidate_path


__all__ = ("setup", "act")
