# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The quadruple tank, declared for drto: the canonical multivariable example.

Johansson's quadruple-tank process (IEEE TCST 8(3), 2000): four tank levels,
two pump flows, each pump splitting between a lower and the diagonal upper
tank (fraction gamma below, 1 - gamma above), Torricelli outflow. With
gamma = 0.4 on both pumps the plant is in its nonminimum-phase
configuration. Absolute levels and flows (not deviations), tracking stage
and terminal costs toward the model's steady state, and initial conditions
(x1_hat..x4_hat). The level setpoints are the exact equilibrium for the pump
setpoints, derived from 0 = f.

Usage from a notebook in ``examples/``::

    from models.quad_tank import quad_tank
    m = quad_tank(N=15, h=10)
"""
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

import drto


def quad_tank(N=15, h=10):
    """Return the declared quadruple tank with an ``N``-step horizon.

    The time set is initialized with the sample grid (``N`` steps of the
    sampling time ``h``, seconds; the default horizon is 150 s). Physical
    constants and setpoints are mutable Params; the initial levels are set
    through ``m.x1_hat`` .. ``m.x4_hat``.
    """
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0, N * h, h))

    m.a1 = pyo.Param(initialize=0.233, mutable=True)  # outlet areas (cm2)
    m.a2 = pyo.Param(initialize=0.242, mutable=True)
    m.a3 = pyo.Param(initialize=0.127, mutable=True)
    m.a4 = pyo.Param(initialize=0.127, mutable=True)
    m.A1 = pyo.Param(initialize=50.27, mutable=True)  # tank cross-sections (cm2)
    m.A2 = pyo.Param(initialize=50.27, mutable=True)
    m.A3 = pyo.Param(initialize=28.27, mutable=True)
    m.A4 = pyo.Param(initialize=28.27, mutable=True)
    m.g = pyo.Param(initialize=981, mutable=True)  # gravity (cm/s2)
    m.gamma = pyo.Param(initialize=0.4, mutable=True)  # pump flow split, both pumps

    # the exact equilibrium for the pump setpoints (0 = f, derived)
    m.x1_ss = pyo.Param(initialize=13.9883, mutable=True)  # level setpoints (cm)
    m.x2_ss = pyo.Param(initialize=14.0644, mutable=True)
    m.x3_ss = pyo.Param(initialize=14.2562, mutable=True)
    m.x4_ss = pyo.Param(initialize=21.4277, mutable=True)
    m.u1_ss = pyo.Param(initialize=43.4, mutable=True)  # pump setpoints (ml/s)
    m.u2_ss = pyo.Param(initialize=35.4, mutable=True)

    m.x1_hat = pyo.Param(initialize=18.9883, mutable=True)  # initial levels: the
    m.x2_hat = pyo.Param(initialize=9.0644, mutable=True)  # setpoints offset by
    m.x3_hat = pyo.Param(initialize=19.2562, mutable=True)  # (+5, -5, +5, -5) cm
    m.x4_hat = pyo.Param(initialize=16.4277, mutable=True)

    m.x1 = pyo.Var(m.t, bounds=(7.5, 28), initialize=13.9883)
    m.x2 = pyo.Var(m.t, bounds=(7.5, 28), initialize=14.0644)
    m.x3 = pyo.Var(m.t, bounds=(3.5, 28), initialize=14.2562)
    m.x4 = pyo.Var(m.t, bounds=(4.5, 28), initialize=21.4277)
    m.dx1 = DerivativeVar(m.x1, wrt=m.t)
    m.dx2 = DerivativeVar(m.x2, wrt=m.t)
    m.dx3 = DerivativeVar(m.x3, wrt=m.t)
    m.dx4 = DerivativeVar(m.x4, wrt=m.t)
    m.u1 = pyo.Var(m.t, bounds=(0, 60), initialize=43.4)
    m.u2 = pyo.Var(m.t, bounds=(0, 60), initialize=35.4)
    # unbounded cost vars: a cost var pinned at a bound drags ipopt
    m.cost = pyo.Var(m.t)
    m.term = pyo.Var()

    @m.Constraint(m.t)
    def x1_ode(m, t):
        return m.dx1[t] == -(m.a1 / m.A1) * pyo.sqrt(2 * m.g * m.x1[t]) + (m.a3 / m.A1) * pyo.sqrt(2 * m.g * m.x3[t]) + (m.gamma / m.A1) * m.u1[t]

    @m.Constraint(m.t)
    def x2_ode(m, t):
        return m.dx2[t] == -(m.a2 / m.A2) * pyo.sqrt(2 * m.g * m.x2[t]) + (m.a4 / m.A2) * pyo.sqrt(2 * m.g * m.x4[t]) + (m.gamma / m.A2) * m.u2[t]

    @m.Constraint(m.t)
    def x3_ode(m, t):
        return m.dx3[t] == -(m.a3 / m.A3) * pyo.sqrt(2 * m.g * m.x3[t]) + ((1 - m.gamma) / m.A3) * m.u2[t]

    @m.Constraint(m.t)
    def x4_ode(m, t):
        return m.dx4[t] == -(m.a4 / m.A4) * pyo.sqrt(2 * m.g * m.x4[t]) + ((1 - m.gamma) / m.A4) * m.u1[t]

    @m.Constraint(sorted(m.t)[:-1])  # the terminal cost owns the final time
    def stage(m, t):
        return m.cost[t] == (m.x1[t] - m.x1_ss) ** 2 + (m.x2[t] - m.x2_ss) ** 2 + (m.x3[t] - m.x3_ss) ** 2 + (m.x4[t] - m.x4_ss) ** 2 + (m.u1[t] - m.u1_ss) ** 2 + (m.u2[t] - m.u2_ss) ** 2

    tN = m.t.last()

    @m.Constraint()  # the stage cost with the controls removed, at tN
    def terminal(m):
        return m.term == (m.x1[tN] - m.x1_ss) ** 2 + (m.x2[tN] - m.x2_ss) ** 2 + (m.x3[tN] - m.x3_ss) ** 2 + (m.x4[tN] - m.x4_ss) ** 2

    @m.Constraint()
    def x1_init(m):
        return m.x1[0] == m.x1_hat

    @m.Constraint()
    def x2_init(m):
        return m.x2[0] == m.x2_hat

    @m.Constraint()
    def x3_init(m):
        return m.x3[0] == m.x3_hat

    @m.Constraint()
    def x4_init(m):
        return m.x4[0] == m.x4_hat

    drto.horizon(m.t)
    drto.state(m.x1, m.x2, m.x3, m.x4)
    drto.dynamics(m.x1_ode, m.x2_ode, m.x3_ode, m.x4_ode)
    drto.control(m.u1, m.u2, profile="piecewise_constant")
    drto.tracking_stage_cost(m.stage)
    drto.tracking_terminal_cost(m.terminal)
    drto.initial_condition(m.x1_init, m.x2_init, m.x3_init, m.x4_init)
    drto.steady_state(m.x1, m.x1_ss)
    drto.steady_state(m.x2, m.x2_ss)
    drto.steady_state(m.x3, m.x3_ss)
    drto.steady_state(m.x4, m.x4_ss)
    drto.steady_state_control(m.u1, m.u1_ss)
    drto.steady_state_control(m.u2, m.u2_ss)
    return m
