"""Exercise notebook definitions without executing demonstration cells."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def definitions(notebook, cell_index, namespace=None):
    cell = json.loads((ROOT / notebook).read_text())["cells"][cell_index]
    tree = ast.parse("".join(cell["source"]))
    classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)], type_ignores=[])
    namespace = {"np": np} if namespace is None else namespace
    exec(compile(classes, str(ROOT / notebook), "exec"), namespace)
    return namespace


class NotebookModels(unittest.TestCase):
    def ring_namespace(self):
        return definitions("RingAttractor.ipynb", 7)

    def test_reset_clears_state_without_destroying_shape_or_weights(self):
        ring = self.ring_namespace()["RingAttractor"](9)
        original_weights = ring.weights.copy()
        ring.v[:] = 2
        ring.spikes[:] = 1
        ring.reset()
        np.testing.assert_array_equal(ring.v, np.zeros(9))
        np.testing.assert_array_equal(ring.spikes, np.zeros(9))
        np.testing.assert_array_equal(ring.weights, original_weights)

    def test_winner_inhibits_neighbors_with_negative_float_values(self):
        ring = self.ring_namespace()["RingAttractor"](9, tau=0, WTA=True)
        ring.v[4] = 2
        ring.step(np.zeros(9))
        np.testing.assert_array_equal(ring.spikes, np.eye(9)[4])
        np.testing.assert_allclose(ring.v, [0, 0, -.3, -.5, 0, -.5, -.3, 0, 0])

    def test_all_controller_variants_support_non_twenty_size(self):
        for cell_index in (34, 47, 49):
            with self.subTest(cell=cell_index):
                namespace = definitions("RingAttractor.ipynb", cell_index, self.ring_namespace())
                controller = namespace["ControllableRing"](9)
                self.assertEqual(controller.left_to_ring.shape, (9, 9))
                self.assertEqual(controller.right_to_ring.shape, (9, 9))
                self.assertEqual(np.count_nonzero(controller.left_to_ring), 27)
                self.assertEqual(np.count_nonzero(controller.right_to_ring), 27)
                if cell_index != 34:
                    controller.step(np.zeros(9))  # No goal configured yet.

    def test_ad_hoc_controller_uses_explicit_input_without_notebook_global(self):
        namespace = definitions("RingAttractor.ipynb", 34, self.ring_namespace())
        controller = namespace["ControllableRing"](9)
        received = []
        controller.ring.step = lambda value: received.append(value.copy())
        controller.left_ring.step = lambda value: None
        controller.right_ring.step = lambda value: None
        expected = np.arange(9, dtype=float)
        controller.step(expected, 0, 0)
        np.testing.assert_array_equal(received[0], expected)

    def test_pid_direction_wraps_using_actual_ring_size(self):
        namespace = definitions("RingAttractor.ipynb", 47, self.ring_namespace())
        controller = namespace["ControllableRing"](9)
        controller.set_goals(8)
        controller.set_pid_controllers(1, 0, 0, 1, 0, 0)
        controller.step(np.zeros(9))
        self.assertEqual(controller.control_errors_left[-1], 0)
        self.assertEqual(controller.control_errors_right[-1], 1)

    def test_visual_direction_uses_circular_angular_distance(self):
        network = definitions("Superior_Colliculus_CAN.ipynb", 1)["NeuralNetwork"](16)
        for point, expected in [((1, 0), 0), ((0, 1), 4), ((-1, 0), 8), ((0, -1), 12)]:
            with self.subTest(point=point):
                self.assertEqual(int(np.argmax(network.stimulate_specific_area(*point))), expected)
        for angle in (-.01, .01):
            self.assertEqual(int(np.argmax(network.stimulate_specific_area(np.cos(angle), np.sin(angle)))), 0)


if __name__ == "__main__":
    unittest.main()
