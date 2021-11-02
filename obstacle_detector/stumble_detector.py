
"""
What defines a "stumble"?
We assume the stumble starts at
-the last point where feet was in contact with ground,
before - the last point where head angle crossed +/- X degrees,
before - the point where the agent fell.
"""
import math
from collections import deque


class StumbleDetector:
    def __init__(self, max_states=1000, angle_threshold=20):
        self.max_states = max_states
        self.radian_threshold = angle_threshold * math.pi / 180
        self.recorded_states = deque()

    def reset_state(self):
        self.recorded_states = deque()

    def add_state(self, pos, angle, left_leg, right_leg):
        self.recorded_states.append((pos, angle, left_leg, right_leg))
        if len(self.recorded_states) > self.max_states:
            self.recorded_states.popleft()

    def find_stumble_pos(self):
        pos, angle, stumble_rl, stumble_ll = self.recorded_states.pop()
        while abs(angle) > self.radian_threshold and len(self.recorded_states) > 0:
            # Angle of hull indicates we are mid stumble
            pos, angle, stumble_rl, stumble_ll = self.recorded_states.pop()

        while not stumble_rl and not stumble_ll and len(self.recorded_states) > 0:
            # Legs are not in contact with ground, so we assume we are mid stumble
            pos, angle, stumble_rl, stumble_ll = self.recorded_states.pop()

        return pos

