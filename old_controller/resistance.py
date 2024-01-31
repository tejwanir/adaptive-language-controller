import numpy as np
from utils import sigmoid


class ResistanceTracker:

    def __init__(self, window_len: int, sensitivity: float, break_point: float):
        self.window_len = window_len
        self.driving_force = None
        self.force_context = [1] * window_len
        self.sensitivity = sensitivity
        self.break_point = break_point
        self.resistance = 0
        self.dt = 1 / 500
    def should_break(self):
        if self.resistance > 0.95:
            return True
        return False

    def avg_force(self):
        return sum(self.force_context) / len(self.force_context)

    def update_resistance(self, external_force):

        alpha = 0.1
        max_force_norm = 15.0
        force_norm = min(max_force_norm, np.linalg.norm(external_force))
        dt = 1 / 500
        adt = alpha**dt
        self.resistance = adt * self.resistance + force_norm / max_force_norm * (1 - adt)
        return self.resistance


