# Continuous attractor networks for navigation

Exploratory Python notebooks for a spiking ring attractor, directional control,
and a visual/superior-colliculus map. These are prototypes, not a validated model
of navigation or a finished scientific result.

## Start here

- [RingAttractor.ipynb](RingAttractor.ipynb): ring dynamics, cue tracking,
  left/right controllers, and exploratory PID variants. Run cells in order in a
  fresh kernel; later cells redefine the controller intentionally.
- [Superior_Colliculus_CAN.ipynb](Superior_Colliculus_CAN.ipynb): a small visual
  direction map and illustrative coordinate transformation.

Both notebooks have Colab links in their first cell. For local use, create a
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

## Check the model definitions

```sh
python -m unittest discover -s tests -v
```

The checks load only class definitions from the actual notebook cells. They do
not replay the long demonstrations, generate new scientific results, or require
a display. They cover reset state, floating-point lateral inhibition, arbitrary
controller size, explicit controller input, and angle wraparound. GitHub Actions
runs the same checks.

Stored notebook outputs have been preserved and were not regenerated after the
code corrections. Re-run relevant cells in a fresh kernel before interpreting
plots as results of the current code.

## Remaining work

- Consolidate the three exploratory `ControllableRing` versions into clearly
  named, importable models after choosing the intended controller behavior.
- Define evaluation metrics and random seeds before comparing tracking or memory
  performance. The current demonstrations alone do not establish stability.
- Decide whether the simple SC transformation is only a visualization or a
  specified biological model before drawing conclusions from it.
