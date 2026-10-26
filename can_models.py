"""Importable model definitions used by RingAttractor.ipynb.

Importing this module performs no simulation, plotting, or random draws. The
three controller variants retain the notebook's original update order and gain
constants; their scientific calibration is not changed by this extraction.
"""

import numpy as np

__all__ = [
    "BrownianThreshold",
    "RingAttractor",
    "PIDController",
    "CurrentControlledRing",
    "HeuristicGoalRing",
    "SqrtPIDGoalRing",
]



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
    """Ring with caller-supplied left and right actuator currents."""

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
    """Goal controller retaining the original heuristic override of PID output."""
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
        if goal < 0 or goal >= self.N:
            raise ValueError("Goal locations must be within the range [0, N-1].")
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
    """Goal controller applying PID to signed square-root position errors."""
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
        if goal < 0 or goal >= self.N:
            raise ValueError("Goal locations must be within the range [0, N-1].")
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
