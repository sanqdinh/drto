# drto

Dynamic real-time optimization: receding-horizon optimization and
estimation for Pyomo models, with advanced-step NMPC as the headline
capability and moving horizon estimation as the planned follow-on.

## The six modes

drto runs one declared model in any of six modes, the 2x3 grid of
{steady-state, dynamic} by {simulation, optimization, estimation}. You
write the model once; the mode fixes what is free and what the objective
is.

|  | Simulation | Optimization | Estimation |
| --- | --- | --- | --- |
| **Steady-state** | solve the model at equilibrium | economic RTO | data reconciliation |
| **Dynamic** | integrate the model forward | NMPC / D-RTO | moving horizon estimation |

Down the columns: simulation frees nothing and solves the model as given;
optimization frees the controls and adds a cost; estimation frees the
states and fits them to measurements. Across the rows: steady-state
collapses the model to a single equilibrium point, dynamic keeps the time
horizon. The optimization and estimation columns are duals (NMPC with MHE,
RTO with reconciliation), so one declaration surface serves both.

The near-term focus is the optimization column: dynamic NMPC/D-RTO, whose
ideal, real-time, and advanced-step execution variants are the headline,
plus steady-state RTO. Estimation is the planned follow-on.

## Declaring a control problem

drto is declaration-first. You write your dynamic model as an ordinary
`pyomo.dae` model, then declare the pieces that turn it into a
receding-horizon control problem (the dynamic-optimization mode); drto
assembles the horizon problem and runs the loop. Those pieces are the six
object types of an optimal control problem:

| DRTO object type | Pyomo object type | Declaration | What it is |
| --- | --- | --- | --- |
| State | Variable | `declare_state(m.z, ...)` | A differential state. drto reads its dynamics from the state's `DerivativeVar`. |
| Control | Variable | `declare_control(m.u, ..., wrt=m.t, profile=...)` | A manipulated input, the decision variable. The `profile` flag sets its parameterization (piecewise-constant, ...) via pyomo-cvp. |
| Stage cost | Constraint | `declare_stage_cost(expr)` | The running cost, summed over the horizon. |
| Terminal cost | Constraint | `declare_terminal_cost(expr)` | The cost on the state at the end of the horizon. |
| Initial condition | Constraint | `declare_initial_condition(...)` | The initial-state anchor, the measurement feedback in NMPC. |
| Terminal constraint | Constraint | `declare_terminal_constraint(...)` | The terminal set or region the final state must lie in. |

Two things you never declare, because they already live in the model: the
**dynamics** are read from the `pyomo.dae` `DerivativeVar`s of the
declared states, and the **path constraints** are the state variables'
own upper and lower bounds.

The vocabulary is the optimal-control literature's own (stage cost,
terminal cost, terminal constraint), so a model reads the way the theory
does. The other modes reuse the same model: simulation drops the cost, and
estimation swaps the initial condition for a soft arrival cost and adds a
measurement.

## Status

Design phase: see [DESIGN.md](DESIGN.md). No code yet.
