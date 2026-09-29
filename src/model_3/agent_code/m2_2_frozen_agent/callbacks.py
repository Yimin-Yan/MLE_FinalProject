"""Expose the trained M2-2 policy as an evaluation-only opponent."""

from agent_code.seat_d3qn.variants.m2_2.callbacks import act, setup


__all__ = ("setup", "act")
