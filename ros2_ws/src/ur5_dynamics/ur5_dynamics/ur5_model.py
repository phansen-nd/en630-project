import numpy as np

class UR5Model:
    def __init__(self):
        self.dh_params = [
            [0,     0.089159,  0,      np.pi/2],
            [0,     0,         -0.425, 0],
            [0,     0,         -0.39225, 0],
            [0,     0.10915,   0,      np.pi/2],
            [0,     0.09465,   0,      -np.pi/2],
            [0,     0.0823,    0,      0]
        ]

    def dh_transform(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])

    def forward_kinematics(self, q):
        assert len(q) == 6
        T = np.eye(4)
        for i in range(6):
            theta = q[i] + self.dh_params[i][0]
            d     = self.dh_params[i][1]
            a     = self.dh_params[i][2]
            alpha = self.dh_params[i][3]
            T = T @ self.dh_transform(theta, d, a, alpha)
        position = T[0:3, 3]
        rotation = T[0:3, 0:3]
        return position, rotation, T

