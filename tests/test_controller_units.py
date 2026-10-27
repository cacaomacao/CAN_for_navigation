"""Small coordinate and interface fixtures; no notebook or trajectory replay."""
import math
import unittest

import numpy as np

from can_models import (
    HeuristicGoalRing,
    SqrtPIDGoalRing,
    angle_to_ring_index,
    ring_index_to_angle,
)


class ControllerUnits(unittest.TestCase):
    def test_reference_angles_and_wrapped_seams(self):
        for N in (20, 40):
            with self.subTest(N=N):
                self.assertAlmostEqual(ring_index_to_angle(N / 4, N), math.pi / 2)
                self.assertAlmostEqual(angle_to_ring_index(-math.pi / 2, N), 3 * N / 4)
                self.assertEqual(ring_index_to_angle(3 * N, N), 0)
                self.assertEqual(angle_to_ring_index(-2 * math.tau, N), 0)
                self.assertLess(angle_to_ring_index(-1e-8, N), N)
                self.assertAlmostEqual(ring_index_to_angle(-1e-8, N), math.tau, places=7)

    def test_continuous_coordinate_round_trip_preserves_fractional_positions(self):
        # Expected positions include fractional values beyond either seam;
        # conversions must not silently select a neuron or clip a whole turn.
        for N in (9, 20, 40):
            for coordinate, expected in ((-.25, N - .25), (.25, .25), (N + .25, .25)):
                with self.subTest(N=N, coordinate=coordinate):
                    angle = ring_index_to_angle(coordinate, N)
                    self.assertGreaterEqual(angle, 0)
                    self.assertLess(angle, math.tau)
                    self.assertAlmostEqual(angle_to_ring_index(angle, N), expected)

    def test_coordinate_helpers_reject_invalid_domains(self):
        for convert in (ring_index_to_angle, angle_to_ring_index):
            for size in (0, -1, 9.5, True):
                with self.subTest(convert=convert.__name__, size=size):
                    with self.assertRaises(ValueError):
                        convert(0, size)
            for coordinate in (math.nan, math.inf, -math.inf, True, 1j):
                with self.subTest(convert=convert.__name__, coordinate=coordinate):
                    with self.assertRaises(ValueError):
                        convert(coordinate, 20)

    def test_goals_are_finite_continuous_indices_and_rejection_preserves_goal(self):
        for variant in (HeuristicGoalRing, SqrtPIDGoalRing):
            controller = variant(20)
            controller.set_goals(19.75)
            controller.set_pid_controllers(1, .2, .3, 1, .2, .3)
            controller.step(np.zeros(20))
            pid_states = [pid.__dict__.copy() for pid in (controller.pid_left, controller.pid_right)]
            log_names = (
                "control_errors_left", "control_errors_right", "control_signals_left",
                "control_signals_right", "cur_spikes",
            )
            logs = {name: getattr(controller, name).copy() for name in log_names}
            arrays = [controller.bump_avg.copy()]
            for ring in (controller.ring, controller.left_ring, controller.right_ring):
                arrays.extend(value.copy() for value in (ring.v, ring.spikes))
            for invalid in (math.nan, math.inf, -math.inf, -.25, 20):
                with self.subTest(variant=variant.__name__, invalid=invalid):
                    with self.assertRaises(ValueError):
                        controller.set_goals(invalid)
                    self.assertEqual(controller.goal, 19.75)
                    self.assertEqual(controller.pid_left.__dict__, pid_states[0])
                    self.assertEqual(controller.pid_right.__dict__, pid_states[1])
                    for name, previous in logs.items():
                        self.assertEqual(getattr(controller, name), previous)
                    current_arrays = [controller.bump_avg]
                    for ring in (controller.ring, controller.left_ring, controller.right_ring):
                        current_arrays.extend((ring.v, ring.spikes))
                    for current, previous in zip(current_arrays, arrays):
                        np.testing.assert_array_equal(current, previous)

    def test_same_quarter_turn_has_different_historical_current_scales(self):
        # Manual 90-degree examples from CONTROLLER_UNITS.md. A silent fixture
        # deliberately uses the existing argmax(all-zero)==0 estimate.
        for N, expected_heuristic, expected_pid, expected_log in (
            (20, .1, .1118033988749895, .0447213595499958),
            (40, .2, .1581138830084190, .0632455532033676),
        ):
            with self.subTest(N=N):
                heuristic = HeuristicGoalRing(N)
                sqrt_pid = SqrtPIDGoalRing(N)
                for controller in (heuristic, sqrt_pid):
                    controller.set_goals(angle_to_ring_index(math.pi / 2, N))
                    controller.set_pid_controllers(1, 0, 0, 1, 0, 0)
                    controller.step(np.zeros(N))
                self.assertEqual(heuristic.control_signals_left[-1], 0)
                self.assertAlmostEqual(heuristic.control_signals_right[-1], expected_heuristic)
                self.assertAlmostEqual(sqrt_pid.pid_left.previous_error, -expected_pid)
                self.assertAlmostEqual(sqrt_pid.pid_right.previous_error, expected_pid)
                self.assertAlmostEqual(sqrt_pid.control_signals_right[-1], expected_pid)
                self.assertAlmostEqual(sqrt_pid.control_errors_right[-1], expected_log)

    def test_antipodal_tie_keeps_negative_branch(self):
        heuristic = HeuristicGoalRing(20)
        sqrt_pid = SqrtPIDGoalRing(20)
        for controller in (heuristic, sqrt_pid):
            controller.set_goals(10)
            controller.set_pid_controllers(1, 0, 0, 1, 0, 0)
            controller.step(np.zeros(20))
        self.assertEqual(heuristic.control_errors_left[-1], 0)
        self.assertEqual(heuristic.control_errors_right[-1], 10)
        self.assertAlmostEqual(heuristic.control_signals_left[-1], .2)
        self.assertGreater(sqrt_pid.control_signals_left[-1], 0)
        self.assertLess(sqrt_pid.control_signals_right[-1], 0)


if __name__ == "__main__":
    unittest.main()
