# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""drto-aware plotting for the example notebooks.

Both functions read everything from the model's registry (``drto.info``): the
declared horizon and its sample grid, the declared states or controls, and
the paired steady-state targets for the dotted setpoint lines. If the model
carries an infinite-horizon terminal segment (``drto_infinite_horizon``), the
tail is found automatically, mapped back to real time through
``t = tN + atanh(tau)/gamma``, and drawn with open markers, squares marking
the segment's finite element boundaries (continuity extrapolations). Points
only, finite horizon filled, tail open, clipped to ``t_max``.
"""
import math

import matplotlib.pyplot as plt
import pyomo.environ as pyo
from matplotlib.lines import Line2D

import drto


def _tail(m):
    """Return (segment block, tN, gamma) or None if no terminal segment."""
    b = m.component("drto_infinite_horizon")
    if b is None:
        return None
    time = drto.info(m).components("horizon")[0]
    return b, time.last(), pyo.value(b.gamma)


def _tail_points(b, tN, gamma, comp, t_max):
    """Map a segment component's points to real time, split at element boundaries.

    Returns (interior, boundary) lists of (t, value) pairs, tau = 1 excluded
    (it maps to t = infinity; its value is the equilibrium endpoint).
    """
    fe = set(b.tau.get_finite_elements())
    interior, boundary = [], []
    for s in sorted(comp.index_set()):
        if not s < 1:
            continue
        t = tN + math.atanh(s) / gamma
        if t > t_max:
            continue
        (boundary if s in fe else interior).append((t, pyo.value(comp[s])))
    return interior, boundary


def _targets(reg, kind):
    """Map each declared owner component to its paired target Param."""
    return [(rec["of"], rec["component"]) for rec in reg.declarations(kind)]


def _draw(m, comps, targets, sample_slice, t_max, boundary_squares):
    reg = drto.info(m)
    samples = reg.declarations("horizon")[0]["samples"][sample_slice]
    tail = _tail(m)
    drew_boundary = False
    for comp in comps:
        values = [pyo.value(comp[t]) for t in samples]
        (line,) = plt.plot(samples, values, "o", label=comp.local_name)
        color = line.get_color()
        for owner, target in targets:
            if owner is comp:
                plt.axhline(pyo.value(target), color=color, linestyle=":")
        if tail is not None:
            b, tN, gamma = tail
            seg = b.component(comp.local_name)
            if seg is not None:
                interior, boundary = _tail_points(b, tN, gamma, seg, t_max)
                if interior:
                    plt.plot(*zip(*interior), "o", mfc="none", color=color, label=f"{comp.local_name} (tail)")
                if boundary and boundary_squares:
                    plt.plot(*zip(*boundary), "s", mfc="none", color=color)
                    drew_boundary = True
    if tail is not None:
        plt.axvline(tail[1], color="grey", linewidth=0.8)
    plt.xlabel("time")
    plt.ylabel("value")
    handles, labels = plt.gca().get_legend_handles_labels()
    if drew_boundary:
        handles.append(Line2D([], [], marker="s", mfc="none", color="grey", linestyle=""))
        labels.append("element boundary")
    plt.legend(handles, labels)
    plt.show()


def plot_states(m, t_max=50):
    """Plot the declared states: finite samples filled, any tail open.

    Setpoint lines come from the ``steady_state`` pairings; squares mark the
    segment's element boundaries, where the state value is the continuity
    extrapolation rather than a collocation point.
    """
    reg = drto.info(m)
    _draw(m, reg.components("state"), _targets(reg, "steady_state"), slice(None), t_max, boundary_squares=True)


def plot_controls(m, t_max=50):
    """Plot the declared controls: one point per move, any tail open.

    The finite points sit at the start of each sampling interval (the final
    sample belongs to the terminal cost, so no move starts there). Setpoint
    lines come from the ``steady_state_control`` pairings. Segment controls
    have no boundary values (a value outside every equation would dangle),
    so no squares.
    """
    reg = drto.info(m)
    _draw(m, reg.components("control"), _targets(reg, "steady_state_control"), slice(None, -1), t_max, boundary_squares=False)
