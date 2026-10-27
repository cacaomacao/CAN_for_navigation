# Controller units and ring size

This contract describes the existing discrete implementation in `can_models.py`.
The `/20`, `/50`, `/100`, gains, update order and notebook outputs are preserved.
The constants are exploratory scales; they do not establish a calibrated PID,
physical current, angular velocity, or equivalent dynamics at different `N`.

## Coordinates and time

`N` is the number of equally spaced index locations in one ring. A full turn is
`N` index steps, `2*pi` radians, or 360 degrees. One index step is `2*pi/N`
radians. Indices increase in the positive angular direction by convention;
the physical zero and clockwise/counterclockwise choice belong to the caller.
The current convolution kernels require `N >= 7` for all three ring variants.

`set_goals(goal)` accepts a finite index in `[0, N)`, including fractional
indices. It does not accept radians or wrap out-of-range goals. A value such as
`pi/2` is a valid numeric index, so unit mistakes cannot be detected from its
value alone. Convert radians explicitly:

```python
from math import pi
from can_models import angle_to_ring_index, ring_index_to_angle

goal = angle_to_ring_index(pi / 2, 20)  # 5.0 index steps
angle = ring_index_to_angle(-1, 20)    # 19*pi/10 radians, wrapped
```

Both helpers accept finite real scalars and positive integer `N`, wrap multiple
turns, and preserve fractional coordinates without rounding. Their outputs are
in `[0, 2*pi)` and `[0, N)` respectively, up to floating-point precision. Helpers
also work for rings smaller than seven; this does not relax the model's kernel
size requirement. They convert positions, not signed errors: multiply a signed
index error by `2*pi/N` directly to retain its sign. If a caller needs a nearest
neuron, its rounding and tie policy must be chosen separately (and wrapped).

One `step` or PID `update` is one discrete tick. There is no seconds-valued `dt`.
`RingAttractor.tau=0.1` is a per-tick decay coefficient, not a time duration;
the goal estimate uses a per-tick moving average `0.95*old + 0.05*spikes`.
`PIDController` uses `I[t] = I[t-1] + u[t]` and `D[t] = u[t] - u[t-1]`, with
both initial states zero. Thus the integral is in input-error times ticks and
the derivative in input-error per tick, with the tick numerically one. Gains
refer to this sample convention. Changing the call rate is not a conversion
to continuous-time PID.

## What reaches each actuator

All currents are additive model input units, the same numerical units as the
external `input` and membrane update. They have no assigned ampere or
radians-per-tick calibration. Scalars broadcast to all neurons; an `N`-element
array can supply spatial input. `CurrentControlledRing.step(input,
left_current, right_current)` accepts those currents directly. It defines no
goal error, PID gains, or implicit scaling by `N`.

The goal variants first step the main ring using the previous side-ring spikes,
then update `bump_avg` and estimate `b = argmax(bump_avg)`. They compute
`d = (goal - b + N) % N`. Define the signed shortest index displacement as
`e = d` when `d < N/2`, otherwise `e = d - N`. Exactly half a turn selects the
negative branch; zero selects the positive branch. These are deterministic
index conventions, not a physical choice of turn direction.

| Quantity | `HeuristicGoalRing` | `SqrtPIDGoalRing` |
| --- | --- | --- |
| Raw side errors | If `e >= 0`: left=`e`, right=0; otherwise left=0, right=`-e`. Units: index steps. | If `e >= 0`: left=`-sqrt(e)`, right=`sqrt(e)`; otherwise left=`sqrt(-e)`, right=`-sqrt(-e)`. Units: square root of index steps. |
| PID input `u` | Raw side errors, without normalization. | Raw side errors `/20`. |
| Applied side currents | Positive `e` drives **right** with `e/50`; negative `e` drives **left** with `-e/50`; the other side is zero. The computed `PID(raw)/100` is discarded, although PID state advances. | PID outputs are applied directly, including negative currents. The integral can continue to drive current when the instantaneous error is zero. |
| `control_errors_left/right` | Raw side errors in index steps. | Raw side errors `/50`, **not the PID inputs**: each logged value is `20/50 = 0.4` times the corresponding PID input. |
| `control_signals_left/right` | Applied heuristic currents after the override. Changing PID gains does not change these currents. | Applied PID currents. |

The left/right error names are historical: a positive index displacement is
recorded as a left error by the heuristic but activates its right current.
Do not infer an anatomical direction from these names. `cur_spikes` records
`argmax` of the current spikes, whereas the error uses `argmax(bump_avg)`;
they can differ. With no activity, an all-zero estimate has `argmax == 0`.
Ties also select the first maximum, so this is not a circular mean or a
confidence-aware position estimate. Before a goal is set, the main ring still
steps, but side-ring updates and PID/log updates in the goal variants are skipped.

## Manual examples for two ring sizes

Suppose the main estimate is index zero and the goal is a quarter turn away.
For the square-root controller use `kp=1, ki=0, kd=0` on both sides and fresh
PID state. These are arithmetic examples, not simulations or tracking results.

| Quantity | `N=20`, goal=5 | `N=40`, goal=10 |
| --- | --- | --- |
| Angular error | `pi/2` (90 degrees) | `pi/2` (90 degrees) |
| Signed index error `e` | 5 | 10 |
| Heuristic raw/logged left error | 5 | 10 |
| Heuristic applied currents (left, right) | `(0, 0.1)` | `(0, 0.2)` |
| Square-root PID inputs (left, right) | `(-sqrt(5)/20, sqrt(5)/20)`, approximately `(-0.1118034, 0.1118034)` | `(-sqrt(10)/20, sqrt(10)/20)`, approximately `(-0.1581139, 0.1581139)` |
| Square-root logged errors (left, right) | approximately `(-0.0447214, 0.0447214)` | approximately `(-0.0632456, 0.0632456)` |
| Square-root applied currents with these gains | Equal to the PID inputs | Equal to the PID inputs |

Wrap example: at `N=20`, estimate 19 and goal 1 give `e=+2`, or 36 degrees.
At `N=40`, estimate 38 and goal 2 give `e=+4`, the same 36 degrees. Reversing
estimate and goal gives `-2` and `-4`. At the antipode (0 to 10 for `N=20`,
0 to 20 for `N=40`) the tie rule gives `-10` and `-20` respectively.

## What scaling by N can and cannot mean

For matching angular error histories and the same tick convention, changing
`N` by a factor `s = N_new/N_old` multiplies index errors by `s` and square-root
index errors by `sqrt(s)`. This yields algebraic conversions, not validated
controller settings:

- Heuristic applied current grows by `s` for the same angle. PID gains cannot
  compensate because the override discards their outputs. Dividing the
  heuristic index-to-current coefficient by `s` would match its instantaneous
  current algebraically; that change is not implemented here.
- The heuristic's discarded PID calculation would match its former numerical
  output if all three gains were divided by `s`, starting with zero or
  consistently converted PID state. This has no effect on the applied current.
- Square-root PID input grows by `sqrt(s)`. With zero or consistently converted
  state, dividing **all** of `kp`, `ki`, `kd` by `sqrt(s)` would match the numerical
  PID outputs for those matching histories. For 20 to 40, that factor is
  `1/sqrt(2)`. Fixed `/20` and `/50` are still not angular normalizations, and the
  stored error logs would still differ by `sqrt(2)`.
- Direct current control has no positional gain to convert. Its mapping from
  current to motion would need an independently specified calibration.

None of these identities guarantees equivalent closed-loop dynamics. Kernel
widths, lateral inhibition, and side-ring offsets are fixed in neuron indices,
so their angular widths change with `N`; winner selection, silence, quantized
estimates, tick rate and accumulated PID state also matter. A physical model,
sampling contract, evaluation metrics and new evidence are required before
choosing calibrated gains or comparing navigation performance. No such
experiment, notebook replay, or new plot is part of this interface contract.
