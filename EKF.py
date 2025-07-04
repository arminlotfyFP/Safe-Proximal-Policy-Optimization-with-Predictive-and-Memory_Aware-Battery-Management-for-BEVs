import numpy as np


class EKF():
    def __init__(self, dt, w,v, x0=None):
        self.dt = dt
        self.w = w
        self.v = v
        self.x = x0 if x0 is not None else np.zeros((2, 1))
        self.P = np.eye(2) * 1e-3