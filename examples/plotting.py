# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""drto-aware plotting for the example notebooks.

Both functions read everything from the model's registry (``drto.info``): the
declared horizon and its sample grid, the declared states or controls, and
the paired steady-state targets for the dotted setpoint lines. If the model
carries an infinite-horizon terminal segment (``drto_infinite_horizon``), the
tail is found automatically, mapped back to real time through
``t = tN + atanh(tau)/gamma``, and drawn with open markers, squares marking
the segment's finite element boundaries (continuity extrapolations).

Each component gets its own fixed-size panel in a two-column grid, all by
default or a selection by name or component. Points only, finite horizon
filled, tail open, clipped to ``t_max``. The functions return the list of
panel axes for further tweaking.
"""
import math

import matplotlib.pyplot as plt
import pyomo.environ as pyo
from matplotlib.lines import Line2D

import drto

#: Fixed panel size (inches): every plot the same size regardless of count.
_PANEL = (5.0, 3.2)


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
    return {id(rec["of"]): rec["component"] for rec in reg.declarations(kind)}


def _select(declared, selection, what):
    """Resolve a selection of names or components against the declared ones."""
    if selection is None:
        return list(declared)
    chosen = []
    for item in selection:
        for comp in declared:
            if comp is item or comp.local_name == item:
                chosen.append(comp)
                break
        else:
            names = ", ".join(c.local_name for c in declared)
            raise ValueError(f"'{item}' is not a declared {what}; declared: {names}.")
    return chosen


def _draw(m, comps, targets, sample_slice, t_max, boundary_squares):
    reg = drto.info(m)
    samples = reg.declarations("horizon")[0]["samples"][sample_slice]
    tail = _tail(m)
    rows = max(1, math.ceil(len(comps) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(2 * _PANEL[0], rows * _PANEL[1]), sharex=True, squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(comps):]:
        ax.axis("off")  # keep the empty slot so every panel stays the same size
    drew_tail = drew_boundary = drew_target = False
    for ax, comp in zip(flat, comps):
        ax.plot(samples, [pyo.value(comp[t]) for t in samples], "o", color="C0")
        target = targets.get(id(comp))
        if target is not None:
            ax.axhline(pyo.value(target), color="C0", linestyle=":")
            drew_target = True
        if tail is not None:
            b, tN, gamma = tail
            seg = b.component(comp.local_name)
            if seg is not None:
                interior, boundary = _tail_points(b, tN, gamma, seg, t_max)
                if interior:
                    ax.plot(*zip(*interior), "o", mfc="none", color="C0")
                    drew_tail = True
                if boundary and boundary_squares:
                    ax.plot(*zip(*boundary), "s", mfc="none", color="C0")
                    drew_boundary = True
            ax.axvline(tN, color="grey", linewidth=0.8)
        ax.set_title(comp.local_name)
    for ax in flat[max(0, len(comps) - 2):len(comps)]:
        ax.set_xlabel("time")
    handles = [Line2D([], [], marker="o", color="C0", linestyle="")]
    labels = ["finite horizon"]
    if drew_tail:
        handles.append(Line2D([], [], marker="o", mfc="none", color="C0", linestyle=""))
        labels.append("tail")
    if drew_boundary:
        handles.append(Line2D([], [], marker="s", mfc="none", color="C0", linestyle=""))
        labels.append("element boundary")
    if drew_target:
        handles.append(Line2D([], [], color="C0", linestyle=":"))
        labels.append("setpoint")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return flat[: len(comps)]


def plot_states(m, states=None, t_max=50):
    """Plot the declared states, one fixed-size panel each, two columns.

    ``states`` selects a subset by name or component (all by default).
    Setpoint lines come from the ``steady_state`` pairings; squares mark the
    segment's element boundaries, where the state value is the continuity
    extrapolation rather than a collocation point. Returns the panel axes.
    """
    reg = drto.info(m)
    comps = _select(reg.components("state"), states, "state")
    return _draw(m, comps, _targets(reg, "steady_state"), slice(None), t_max, boundary_squares=True)


def plot_controls(m, controls=None, t_max=50):
    """Plot the declared controls, one fixed-size panel each, two columns.

    ``controls`` selects a subset by name or component (all by default). The
    finite points sit at the start of each sampling interval (the final
    sample belongs to the terminal cost, so no move starts there). Setpoint
    lines come from the ``steady_state_control`` pairings. Segment controls
    have no boundary values, so no squares. Returns the panel axes.
    """
    reg = drto.info(m)
    comps = _select(reg.components("control"), controls, "control")
    return _draw(m, comps, _targets(reg, "steady_state_control"), slice(None, -1), t_max, boundary_squares=False)
