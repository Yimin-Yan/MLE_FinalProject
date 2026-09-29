"""Expose the trained M2-1 policy as an evaluation-only opponent."""

from agent_code.seat_d3qn.variants.m2_1.callbacks import act, setup


__all__ = ("setup", "act")
