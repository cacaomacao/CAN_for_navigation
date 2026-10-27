"""Importable model definitions used by RingAttractor.ipynb.

Importing this module performs no simulation, plotting, or random draws. The
three controller variants retain the notebook's original update order and gain
constants; their scientific calibration is not changed by this extraction.
"""

import math
from numbers import Integral, Real

import numpy as np

__all__ = [
    "BrownianThreshold",
    "RingAttractor",
    "PIDController",
    "CurrentControlledRing",
    "HeuristicGoalRing",
    "SqrtPIDGoalRing",
    "ring_index_to_angle",
    "angle_to_ring_index",
]


def _coordinate_and_size(value, N):
    if isinstance(N, bool) or not isinstance(N, Integral) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Coordinate must be a finite real scalar.")
    return float(value), int(N)


def ring_index_to_angle(index, N):
    """Wrap a continuous ring index to radians in [0, 2*pi).

    Index zero is angle zero; increasing index increases angle. No rounding,
    controller update, or choice of a physical zero/direction is performed.
    N must be a positive integer and index a finite real scalar.
    """
    index, N = _coordinate_and_size(index, N)
    return ((index % N) / N * math.tau) % math.tau


def angle_to_ring_index(angle, N):
    """Wrap radians to a continuous ring index in [0, N), without rounding.

    N must be a positive integer and angle a finite real scalar. Fractional
    indices can be passed to set_goals; choosing a discrete neuron is separate.
    """
    angle, N = _coordinate_and_size(angle, N)
    return ((angle % math.tau) / math.tau * N) % N



class BrownianThreshold:
  def __init__(self, N, base, tau = 0.1, alpha= 0.02):
    self._threshold = base * np.ones((N))

    self.alpha = alpha
    self.tau = tau
    self.base = base * np.ones((N))
    self.N = N

  def step(self):
    self._threshold += self.tau * (self.base - self._threshold) + self.alpha * np.random.rand(self.N)

  @property
  def threshold(self):
    return self._threshold


class RingAttractor:
    """
        Model for the Ring attractor. Based on a LIF neural Network.
    """
    def __init__(self, N, threshold=1, tau= 0.1, WTA = False):
        """
        Params:
        - N (Int): Number of neurons
        - threshold (float): Threshold value to spike
        - tau (float): decaying factor for the network.
        - WTA (Bool): Whether to use WTA or not.
        """
        self.N = N

        self.v = np.zeros((N))
        self.spikes = np.zeros((N))

        self.weights = np.zeros((N, N))

        self.threshold = threshold
        self.tau = tau

        self.WTA = WTA

        self.simple_init()

    def simple_init(self):
        """ Default parameters for the ring network on the project. """
        self.init_local_connections([-0.4,-0.3,0.6,1,0.6,-0.3,-0.4])

    def reset(self):
        """ convert's all of numbers to zero """
        self.v = np.zeros_like(self.v)
        self.spikes = np.zeros_like(self.spikes)

    def init_local_connections(self, weight_conv):
        """
        Weight initialization process. Creates a diagonal-like matrix with the
        specified weight convolution for each row.
        """
        assert self.N >= len(weight_conv)

        for i, w in enumerate(weight_conv):
            for j in range(self.N):
                self.weights[(i+j-len(weight_conv)//2+self.N)%self.N,(j)%self.N] = w

    def step(self, input):
        """
        Step of the network.
        """

        self.v += -self.tau*self.v + input + self.weights @ self.spikes

        spike_mask = self.v > self.threshold

        if spike_mask.astype(float).sum()>0:
        # Spiking Dynamics

            if self.WTA:
                max_neuron_idx = np.random.choice(np.where(self.v == self.v.max())[0])
                spike_mask[:] = False
                spike_mask[max_neuron_idx] = True

                spike_maks_float = np.zeros_like(spike_mask, dtype=float)
                spike_maks_float[(max_neuron_idx-2+self.N)%self.N] -= 0.3
                spike_maks_float[(max_neuron_idx-1+self.N)%self.N] -= 0.5
                spike_maks_float[(max_neuron_idx+1+self.N)%self.N] -= 0.5
                spike_maks_float[(max_neuron_idx+2+self.N)%self.N] -= 0.3

                self.v += spike_maks_float

            self.v[spike_mask] = 0
        self.spikes = spike_mask.astype(float)


class PIDController:
    """Discrete PID: one update is one tick; there is no physical dt.

    Integral sums input samples and derivative differences consecutive samples.
    Gains therefore refer to the caller's error units and a unit update tick.
    """
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def update(self, error):
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative
        return control_signal


class CurrentControlledRing:
    """Ring with caller-supplied currents in model input units per step.

    No positional error, angular velocity, PID, or scaling by N is inferred.
    """

    def __init__(self, N):
        self.ring = RingAttractor(N, WTA = True)
        self.ring.init_local_connections([-0.4,-0.3,0.6,1,0.6,-0.3,-0.4])

        self.left_ring = RingAttractor(N, WTA = True)
        self.left_ring.init_local_connections([-0.8,-0.6,-0.1,-0.05,-0.1,-0.6,-0.8])

        self.ring_to_left = 0.2 * np.identity(N)
        self.left_to_ring = np.zeros_like(self.ring_to_left)
        for j in range(N):
            self.left_to_ring[j][(j+1)%N] = 0.3
            self.left_to_ring[j][(j+2)%N] = 0.7
            self.left_to_ring[j][(j+3)%N] = 0.3


        self.right_ring = RingAttractor(N, WTA = True)
        self.right_ring.init_local_connections([-0.8,-0.6,-0.1,-0.05,-0.1,-0.6,-0.8])

        self.ring_to_right = 0.2 * np.identity(N)
        self.right_to_ring = np.zeros_like(self.ring_to_right)
        for j in range(N):
            self.right_to_ring[j][(j-1+N)%N] = 0.3
            self.right_to_ring[j][(j-2+N)%N] = 0.7
            self.right_to_ring[j][(j-3+N)%N] = 0.3

    def step(self, input, left_current, right_current):
        self.ring.step(input + self.left_to_ring @ self.left_ring.spikes + self.right_to_ring @ self.right_ring.spikes )

        self.left_ring.step(self.ring_to_left @ self.ring.spikes + left_current)
        self.right_ring.step(self.ring_to_right @ self.ring.spikes + right_current)


class HeuristicGoalRing:
    """Goal in ring-index units; heuristic current is shortest distance / 50.

    PID receives index distances and advances, but its /100 output is replaced.
    See CONTROLLER_UNITS.md for direction, tie, time and scaling conventions.
    """
    def __init__(self, N):
        self.N = N
        self.ring = RingAttractor(N, WTA=True)
        self.ring.init_local_connections([-0.4, -0.3, 0.6, 1, 0.6, -0.3, -0.4])

        self.left_ring = RingAttractor(N, WTA=True)
        self.left_ring.init_local_connections([-0.8, -0.6, -0.1, -0.05, -0.1, -0.6, -0.8])

        self.ring_to_left = 0.2 * np.identity(N)
        self.left_to_ring = np.zeros_like(self.ring_to_left)
        for j in range(N):
            self.left_to_ring[j][(j + 1) % N] = 0.3
            self.left_to_ring[j][(j + 2) % N] = 0.7
            self.left_to_ring[j][(j + 3) % N] = 0.3

        self.right_ring = RingAttractor(N, WTA=True)
        self.right_ring.init_local_connections([-0.8, -0.6, -0.1, -0.05, -0.1, -0.6, -0.8])

        self.ring_to_right = 0.2 * np.identity(N)
        self.right_to_ring = np.zeros_like(self.ring_to_right)
        for j in range(N):
            self.right_to_ring[j][(j - 1 + N) % N] = 0.3
            self.right_to_ring[j][(j - 2 + N) % N] = 0.7
            self.right_to_ring[j][(j - 3 + N) % N] = 0.3

        self.goal = None
        self.goal_left = None
        self.goal_right = None
        self.pid_left = None
        self.pid_right = None

        self.control_signals_left = []
        self.control_signals_right = []

        self.control_errors_left = []
        self.control_errors_right = []

        self.cur_spikes = []

        self.bump_avg = np.zeros((N))

    def set_goals(self, goal):
        """Set a finite ring index in [0, N); fractional indices are allowed."""
        if not np.isfinite(goal) or goal < 0 or goal >= self.N:
            raise ValueError("Goal must be a finite ring index in [0, N).")
        self.goal = goal

    def set_pid_controllers(self, kp_left, ki_left, kd_left, kp_right, ki_right, kd_right):
        self.pid_left = PIDController(kp_left, ki_left, kd_left)
        self.pid_right = PIDController(kp_right, ki_right, kd_right)

    def step(self, input):
        # Calculate the current positions of the left and right rings
        left_current = self.left_to_ring @ self.left_ring.spikes
        right_current = self.right_to_ring @ self.right_ring.spikes

        # Call the step method of the main ring attractor with updated inputs
        self.ring.step(input + left_current + right_current)

        self.bump_avg *= 0.95
        self.bump_avg += 0.05 * self.ring.spikes


        # Calculate the errors between the current position and the goal positions for left and right rings
        if self.goal is not None:

            goal_temp = (self.goal - np.argmax(self.bump_avg) + self.ring.N) % self.N

            if goal_temp < self.N / 2:
                error_left = goal_temp
                error_right = 0
            else:
                error_right = self.N - goal_temp
                error_left = 0

            # Update the PID controllers and get the control signals for left and right rings
            control_signal_left = self.pid_left.update(error_left)/100
            control_signal_right = self.pid_right.update(error_right)/100


            # Preserve the exploratory override: PID state advances, but the
            # actuator currents below replace its output.
            if error_left > 0:
                control_signal_left = 0.0
                control_signal_right = error_left/50
            else:
                control_signal_left = error_right/50
                control_signal_right = 0.0

            self.control_errors_left.append(error_left)
            self.control_errors_right.append(error_right)

            self.control_signals_left.append(control_signal_left)
            self.control_signals_right.append(control_signal_right)

            self.cur_spikes.append(np.argmax(self.ring.spikes))

            # Apply the control signals to move the left and right rings towards their respective goals
            self.left_ring.step(self.ring_to_left @ self.ring.spikes + control_signal_left)
            self.right_ring.step(self.ring_to_right @ self.ring.spikes + control_signal_right)


class SqrtPIDGoalRing:
    """PID input is signed sqrt(index distance) / 20; logs use /50.

    These historical scales do not normalize radians or ring size N.
    See CONTROLLER_UNITS.md for direction, tie, time and scaling conventions.
    """
    def __init__(self, N):
        self.N = N
        self.ring = RingAttractor(N, WTA=True)
        self.ring.init_local_connections([-0.4, -0.3, 0.6, 1, 0.6, -0.3, -0.4])

        self.left_ring = RingAttractor(N, WTA=True)
        self.left_ring.init_local_connections([-0.8, -0.6, -0.1, -0.05, -0.1, -0.6, -0.8])

        self.ring_to_left = 0.2 * np.identity(N)
        self.left_to_ring = np.zeros_like(self.ring_to_left)
        for j in range(N):
            self.left_to_ring[j][(j + 1) % N] = 0.3
            self.left_to_ring[j][(j + 2) % N] = 0.7
            self.left_to_ring[j][(j + 3) % N] = 0.3

        self.right_ring = RingAttractor(N, WTA=True)
        self.right_ring.init_local_connections([-0.8, -0.6, -0.1, -0.05, -0.1, -0.6, -0.8])

        self.ring_to_right = 0.2 * np.identity(N)
        self.right_to_ring = np.zeros_like(self.ring_to_right)
        for j in range(N):
            self.right_to_ring[j][(j - 1 + N) % N] = 0.3
            self.right_to_ring[j][(j - 2 + N) % N] = 0.7
            self.right_to_ring[j][(j - 3 + N) % N] = 0.3

        self.goal = None
        self.goal_left = None
        self.goal_right = None
        self.pid_left = None
        self.pid_right = None

        self.control_signals_left = []
        self.control_signals_right = []

        self.control_errors_left = []
        self.control_errors_right = []

        self.cur_spikes = []

        self.bump_avg = np.zeros((N))

    def set_goals(self, goal):
        """Set a finite ring index in [0, N); fractional indices are allowed."""
        if not np.isfinite(goal) or goal < 0 or goal >= self.N:
            raise ValueError("Goal must be a finite ring index in [0, N).")
        self.goal = goal

    def set_pid_controllers(self, kp_left, ki_left, kd_left, kp_right, ki_right, kd_right):
        self.pid_left = PIDController(kp_left, ki_left, kd_left)
        self.pid_right = PIDController(kp_right, ki_right, kd_right)

    def step(self, input):
        # Calculate the current positions of the left and right rings
        left_current = self.left_to_ring @ self.left_ring.spikes
        right_current = self.right_to_ring @ self.right_ring.spikes

        # Call the step method of the main ring attractor with updated inputs
        self.ring.step(input + left_current + right_current)

        self.bump_avg *= 0.95
        self.bump_avg += 0.05 * self.ring.spikes


        # Calculate the errors between the current position and the goal positions for left and right rings
        if self.goal is not None:

            goal_temp = (self.goal - np.argmax(self.bump_avg) + self.ring.N) % self.N

            if goal_temp < self.N / 2:
                error_left = -np.sqrt(goal_temp)
                error_right = np.sqrt(goal_temp)
            else:
                error_right = -np.sqrt(self.N - goal_temp)
                error_left = np.sqrt(self.N - goal_temp)

            # Update the PID controllers and get the control signals for left and right rings
            # Historical scales are intentionally fixed, even when N != 20.
            control_signal_left = self.pid_left.update(error_left/20)
            control_signal_right = self.pid_right.update(error_right/20)

            self.control_errors_left.append(error_left/50)
            self.control_errors_right.append(error_right/50)

            self.control_signals_left.append(control_signal_left)
            self.control_signals_right.append(control_signal_right)

            self.cur_spikes.append(np.argmax(self.ring.spikes))

            # Apply the control signals to move the left and right rings towards their respective goals
            self.left_ring.step(self.ring_to_left @ self.ring.spikes + control_signal_left)
            self.right_ring.step(self.ring_to_right @ self.ring.spikes + control_signal_right)
