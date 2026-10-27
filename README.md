# Continuous attractor networks for navigation

Exploratory Python notebooks for a spiking ring attractor, directional control,
and a visual/superior-colliculus map. These are prototypes, not a validated model
of navigation or a finished scientific result.

## Start here

- [RingAttractor.ipynb](RingAttractor.ipynb): ring dynamics, cue tracking,
  left/right controllers, and exploratory PID variants. Run cells in order in a
  fresh kernel. Each controller section imports an explicitly named variant.
- [Superior_Colliculus_CAN.ipynb](Superior_Colliculus_CAN.ipynb): a small visual
  direction map and illustrative coordinate transformation.

Both notebooks have Colab links in their first cell. For local use, download or
clone the complete repository, change into its root directory, and create a
project environment with Python 3.11 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab
```

NumPy and Matplotlib are the runtime dependencies. PyTorch is not used by these
notebooks. Dependency ranges are recorded; an exact environment lock and seeded
end-to-end scientific reproduction are still future work.

### Colab and standalone notebook copies

`RingAttractor.ipynb` imports [can_models.py](can_models.py). For Colab, download
that file from **the same branch/commit as the notebook**, then upload it using
Colab's Files panel into the runtime's working directory (normally `/content`)
before running the model import cells. After a runtime reset, upload it again.
For a local standalone notebook, place the matching file beside the notebook.
Alternatively, open the notebook from a complete checkout. The import cells do
not download code automatically. `Superior_Colliculus_CAN.ipynb` is unchanged
and remains self-contained.

## Use the models without notebooks

From the repository root, or with `can_models.py` on your Python path:

```python
from can_models import (
    BrownianThreshold, RingAttractor, PIDController,
    CurrentControlledRing, HeuristicGoalRing, SqrtPIDGoalRing,
)

current_control = CurrentControlledRing(20)
heuristic_control = HeuristicGoalRing(20)
sqrt_pid_control = SqrtPIDGoalRing(20)
```

Importing the module only defines classes and coordinate helpers. It uses NumPy
and does not import Matplotlib, consume random draws, run a demonstration, or create plots. Plotting
helpers and demonstration loops remain in the notebook.

| Class | Original notebook section | Control convention |
| --- | --- | --- |
| `CurrentControlledRing` | First `ControllableRing` definition | `step(input, left_current, right_current)` takes actuator currents directly. |
| `HeuristicGoalRing` | First goal/PID experiment | Calls PID, then overwrites its output with the original heuristic currents (`error / 50`). PID state still advances. |
| `SqrtPIDGoalRing` | Second goal/PID experiment | Feeds signed square-root error divided by `20` into PID; logs that error divided by `50`. |

For either goal controller, call `set_pid_controllers(...)` and `set_goals(...)`
before stepping toward a goal. The original ambiguous name `ControllableRing`
is not exported: choose a variant explicitly. Existing demonstrations now do so.

Connection sizes and wraparound use `N`; the convolution kernel requires at
least seven neurons. The historical `/100` PID calculation followed by the
heuristic override, `/20` square-root PID input, and `/50` scales are preserved.
They are fixed experimental constants, **not** a validated normalization for
arbitrary `N`. [Controller units and scaling](CONTROLLER_UNITS.md) defines the
existing index, current and tick conventions, gives manual examples for `N=20`
and `N=40`, and separates algebraic gain conversion from scientific calibration.
For explicit coordinate conversion, `ring_index_to_angle(index, N)` returns
wrapped radians and `angle_to_ring_index(angle, N)` returns a wrapped continuous
index. Neither helper changes controller behavior or rounds to a neuron.

## Check the model definitions

```sh
python -m unittest discover -s tests -v
```

The checks import the ring models directly and load only the SC class definition
from its notebook. They do not replay the long demonstrations, generate new
scientific results, or require a display. The original six regressions retain
their coverage of reset state, floating-point lateral inhibition, non-twenty
controller size, explicit controller input, and angle wraparound. Additional
checks cover standalone imports, distinct notebook bindings, the heuristic
override, coordinate seams, finite goals, and the preserved square-root PID
scales at different ring sizes. GitHub Actions runs the
same checks.

Stored notebook outputs have been preserved and were not regenerated after the
code corrections. Re-run relevant cells in a fresh kernel before interpreting
plots as results of the current code.

## Remaining work

- Validate any proposed angular gain normalization across `N` before comparing
  the three preserved variants. The documented units describe the current
  implementation; no preferred controller or calibrated gains have been selected.
- Define evaluation metrics and random seeds before comparing tracking or memory
  performance. The current demonstrations alone do not establish stability.
- Decide whether the simple SC transformation is only a visualization or a
  specified biological model before drawing conclusions from it.
