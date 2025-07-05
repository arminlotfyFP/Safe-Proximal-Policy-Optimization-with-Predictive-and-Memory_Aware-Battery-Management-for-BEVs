import numpy as np

class BatteryEKF:
    def __init__(self, initial_SOC=0.9, batt_vMAX=3.2, batt_vMIN=2.0, Q_nominal=3600, R0=0.01, C1=2000, R1=0.015, dt=1.0):
        self.dt = dt
        self.Q_nominal = Q_nominal  # Fixed capacity (Coulombs)
        self.R0 = R0
        self.C1 = C1
        self.R1 = R1
        self.SOC_initial = initial_SOC
        self.bat_min = batt_vMIN
        self.bat_max = batt_vMAX

        # Initial states: [SOC, V_RC]
        self.x = np.array([self.SOC_initial, 0.0])  # Initial SOC, V_RC

        # Covariance matrix (2x2 now)
        self.P = np.eye(2) * 0.01

        # Process noise (2x2)
        self.Q_cov = np.diag([1e-5, 1e-4])

        # Measurement noise
        self.R_cov = np.array([[0.01]])

    def ocv(self, soc):
        return 3.0 + abs(self.bat_min - self.bat_max) * soc

    def ocv_derivative(self, soc):
        return abs(self.bat_min - self.bat_max)

    def predict(self, I):
        soc, V_RC = self.x
        dt = self.dt

        # Euler update for SOC
        soc -= (I * dt) / self.Q_nominal

        # Euler update for V_RC
        dV_RC = (-V_RC / (self.R1 * self.C1) + I / self.C1) * dt
        V_RC += dV_RC

        self.x = np.array([soc, V_RC])

        # Jacobian of f wrt x (2x2)
        dfdx = np.array([
            [1, 0],
            [0, 1 - dt / (self.R1 * self.C1)]
        ])

        self.P = dfdx @ self.P @ dfdx.T + self.Q_cov

    def update(self, I, V_measured):
        soc, V_RC = self.x

        # Predicted terminal voltage
        V_pred = self.ocv(soc) - I * self.R0 - V_RC

        # Jacobian of h wrt x (1x2)
        H = np.array([[self.ocv_derivative(soc), -1]])

        # Kalman Gain
        S = H @ self.P @ H.T + self.R_cov
        K = self.P @ H.T @ np.linalg.inv(S)

        # Innovation and update
        y = V_measured - V_pred
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(2) - K @ H) @ self.P

    def step(self, I, V_measured):
        self.predict(I)
        self.update(I, V_measured)
        return self.x.copy()  # [SOC, V_RC]
