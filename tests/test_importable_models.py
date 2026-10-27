"""Portable imports, notebook wiring and the distinct controller conventions."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from can_models import CurrentControlledRing, HeuristicGoalRing, SqrtPIDGoalRing

ROOT = Path(__file__).resolve().parents[1]


class ImportableModels(unittest.TestCase):
    def test_standalone_model_import_needs_no_notebook_or_plotting(self):
        # This is also the Colab upload layout: one model file in the working
        # directory, with NumPy available and no repository checkout required.
        script = """
import sys
import numpy as np
before = np.random.get_state()
import can_models
after = np.random.get_state()
assert 'matplotlib' not in sys.modules
assert before[0] == after[0]
np.testing.assert_array_equal(before[1], after[1])
assert before[2:] == after[2:]
class_names = (
    'BrownianThreshold', 'RingAttractor', 'PIDController',
    'CurrentControlledRing', 'HeuristicGoalRing', 'SqrtPIDGoalRing',
)
for name in class_names:
    assert isinstance(getattr(can_models, name), type), name
for name in ('ring_index_to_angle', 'angle_to_ring_index'):
    assert name in can_models.__all__
    assert callable(getattr(can_models, name)), name
"""
        with tempfile.TemporaryDirectory() as directory:
            shutil.copyfile(ROOT / "can_models.py", Path(directory) / "can_models.py")
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=directory,
                capture_output=True, text=True, timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_notebook_controller_sections_keep_distinct_bindings(self):
        notebook = json.loads((ROOT / "RingAttractor.ipynb").read_text())
        namespace = {}
        # Execute only model imports; no constructors or demonstration cells.
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            tree = ast.parse(source)
            if tree.body and all(
                isinstance(node, ast.ImportFrom) and node.module == "can_models"
                for node in tree.body
            ):
                exec(compile(tree, "RingAttractor.ipynb imports", "exec"), namespace)
        for variant in (CurrentControlledRing, HeuristicGoalRing, SqrtPIDGoalRing):
            self.assertIs(namespace[variant.__name__], variant)
        self.assertNotIn("ControllableRing", namespace)
        # The three demonstrations must choose their own imported variant.
        demo_variants = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                    continue
                if any(isinstance(target, ast.Name) and target.id in ("c_ring", "controllable_ring") for target in node.targets):
                    demo_variants.append(node.value.func.id)
        self.assertEqual(demo_variants, [
            "CurrentControlledRing", "HeuristicGoalRing", "SqrtPIDGoalRing",
        ])

    def test_heuristic_variant_keeps_pid_state_but_overrides_its_output(self):
        controller = HeuristicGoalRing(9)
        controller.set_goals(4)
        controller.set_pid_controllers(3, .2, .5, 3, .2, .5)
        controller.step(np.zeros(9))
        self.assertEqual(controller.pid_left.integral, 4)
        self.assertEqual(controller.control_errors_left[-1], 4)
        self.assertEqual(controller.control_errors_right[-1], 0)
        self.assertEqual(controller.control_signals_left[-1], 0)
        self.assertAlmostEqual(controller.control_signals_right[-1], .08)

    def test_sqrt_pid_keeps_fixed_scaling_for_both_directions(self):
        # N=9 makes an accidental switch from the historical /20 to /N visible.
        for goal, sign in ((4, 1), (5, -1)):
            with self.subTest(goal=goal):
                controller = SqrtPIDGoalRing(9)
                controller.set_goals(goal)
                controller.set_pid_controllers(3, .2, .5, 3, .2, .5)
                controller.step(np.zeros(9))
                self.assertAlmostEqual(controller.pid_left.integral, -.1 * sign)
                self.assertAlmostEqual(controller.pid_right.integral, .1 * sign)
                self.assertAlmostEqual(controller.control_errors_left[-1], -.04 * sign)
                self.assertAlmostEqual(controller.control_errors_right[-1], .04 * sign)
                self.assertAlmostEqual(controller.control_signals_left[-1], -.37 * sign)
                self.assertAlmostEqual(controller.control_signals_right[-1], .37 * sign)


if __name__ == "__main__":
    unittest.main()
