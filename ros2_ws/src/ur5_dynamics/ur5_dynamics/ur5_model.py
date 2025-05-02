import numpy as np
from scipy.spatial.transform import Rotation as R

class UR5Model:
    def __init__(self):
        # From UR official website
        self.dh_params = [
            [0,     0.089159,  0,       np.pi/2 ],
            [0,     0,         0.425,   0       ],
            [0,     0,         0.39225, 0       ],
            [0,     0.10915,   0,       np.pi/2 ],
            [0,     0.09465,   0,       -np.pi/2],
            [0,     0.0823,    0,       0       ]
        ]

    def dh_transform(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])

    def forward_kinematics(self, q):
        T = np.eye(4)
        for i in range(6):
            theta = q[i] + self.dh_params[i][0]
            d     = self.dh_params[i][1]
            a     = self.dh_params[i][2]
            alpha = self.dh_params[i][3]
            T = T @ self.dh_transform(theta, d, a, alpha)
        position = T[0:3, 3]
        rotation = T[0:3, 0:3]
        return position, rotation

    def jacobian(self, q):
        T = np.eye(4)
        origins = [T[0:3, 3]] # o_0
        z_axes = [T[0:3, 2]] # z_0

        # Forward kinematics to collect all o_i and z_i
        for i in range(6):
            theta = q[i] + self.dh_params[i][0]
            d     = self.dh_params[i][1]
            a     = self.dh_params[i][2]
            alpha = self.dh_params[i][3]
            A_i = self.dh_transform(theta, d, a, alpha)
            T = T @ A_i

            origins.append(T[0:3, 3]) # o_i
            z_axes.append(T[0:3, 2]) # z_i

        o_n = origins[-1]
        J = np.zeros((6, 6))

        # Compute each link's current linear/angular Jacobian
        for i in range(6):
            z = z_axes[i]
            o_i = origins[i]
            J_v = np.cross(z, o_n - o_i)
            J_w = z
            J[0:3, i] = J_v
            J[3:6, i] = J_w

        return J

    # Contain the logic for comparing rot matrix w/ quaternion
    def orientation_error(self, R_current, q_goal):
        r_current = R.from_matrix(R_current)
        q_current = r_current.as_quat()
        q_goal = q_goal / np.linalg.norm(q_goal)
        r_goal = R.from_quat(q_goal)
        r_err = r_goal * r_current.inv()
        return r_err.as_rotvec()

    # Jacobian-based IK
    def inverse_kinematics(self, q_init, goal_position, goal_orientation):
        q = q_init.copy()

        for i in range(100):
            current_position, current_rotation = self.forward_kinematics(q)

            e_p = goal_position - current_position
            e_o = self.orientation_error(current_rotation, goal_orientation)
            error = np.concatenate([e_p, e_o])

            if np.linalg.norm(error) < 1e-4:
                break

            q += 0.1 * np.linalg.pinv(self.jacobian(q)) @ error

        return q
