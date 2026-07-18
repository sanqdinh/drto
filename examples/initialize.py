# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""Initialization helper for the double column example.

Interpolates each state linearly in time from its initial condition to its
steady-state target, then fills each algebraic variable from its own
defining equation at those states, member by member, so the solve starts
from a consistent point instead of the flat construction values. No physics
lives here: the algebraic values are computed from the model's equations
with ``calculate_variable_from_constraint``.

A stand-in: initialization proper is a future drto feature. Call after all
transforms, right before the solve; the ``drto.infinite_horizon`` segment
block is handled too when present (its copies hold the horizon-end values,
the steady state, which is where the tail should sit).

Usage from a notebook in ``examples/``::

    from initialize import initialize_column
    initialize_column(m)
"""
import pyomo.environ as pyo
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# state family -> its initial-condition and steady-state Params
_STATES = [
    ("x1", "x1_hat", "x1_ss"),
    ("x2", "x2_hat", "x2_ss"),
    ("M1", "M1_hat", "M1_ss"),
    ("M2", "M2_hat", "M2_ss"),
]

# variable family -> the equation family that defines it, member for member
_PAIRS = [
    ("L1", "L1_def"),
    ("L2", "L2_def"),
    ("y1", "y1_def"),
    ("y2", "y2_def"),
    ("TC1", "TC1_def"),
    ("TC2", "TC2_def"),
    ("purA1", "purA1_def"),
    ("purB2", "purB2_def"),
    ("purC2", "purC2_def"),
    ("cost", "stage"),
    ("term", "terminal"),
]


def initialize_column(m):
    """Interpolate the states and compute the algebraics from the equations."""
    _interpolate_states(m)
    _solve_pairs(m)
    seg = m.component("drto_infinite_horizon")
    if seg is not None:
        _solve_pairs(seg)


def _interpolate_states(m):
    tN = m.t.last()
    for var_name, hat_name, ss_name in _STATES:
        var = m.component(var_name)
        hat, ss = m.component(hat_name), m.component(ss_name)
        for idx in var:
            other, t = idx[:-1], idx[-1]
            a, b = pyo.value(hat[other]), pyo.value(ss[other])
            var[idx].set_value(a + (t / tN) * (b - a))


def _solve_pairs(blk):
    for var_name, con_name in _PAIRS:
        var, con = blk.component(var_name), blk.component(con_name)
        if var is None or con is None:  # the segment carries no cost vars
            continue
        for cd in con.values():
            calculate_variable_from_constraint(var[cd.index()], cd)
