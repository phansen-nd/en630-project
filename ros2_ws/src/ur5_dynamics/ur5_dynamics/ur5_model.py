import numpy as np
from scipy.spatial.transform import Rotation as R

class UR5Model:
    def __init__(self):
        self.dh_params = [
            [0,     0.089159,  0,       np.pi/2 ],
            [0,     0,         0.425,   0       ],
            [0,     0,         0.39225, 0       ],
            [0,     0.10915,   0,       np.pi/2 ],
            [0,     0.09465,   0,       -np.pi/2],
            [0,     0.0823,    0,       0       ]
        ]

        self.link_masses = [3.7, 8.393, 2.33, 1.219, 1.219, 0.1879]
        self.link_com_positions = [
            np.array([0.0, -0.02561, 0.00193]),
            np.array([0.2125, 0.0, 0.11336]),
            np.array([0.15, 0.0, 0.0265]),
            np.array([0.0, -0.0018, 0.01634]),
            np.array([0.0, 0.0018, 0.01634]),
            np.array([0.0, 0.0, -0.001159])
        ]
        self.gravity = np.array([0.0, 0.0, -9.81])

    def mass_matrix(self, q):
        # TODO: incorporate q for more accurate matrix
        return np.diag(self.link_masses)

    def gravity_torques(self, q):
        """
        Compute gravity torques G(q) by summing over each link's COM torque contribution.
        """
        assert len(q) == 6

        G = np.zeros(6)
        T = np.eye(4)

        for i in range(6):
            # Forward kinematics to link i
            theta = q[i] + self.dh_params[i][0]
            d     = self.dh_params[i][1]
            a     = self.dh_params[i][2]
            alpha = self.dh_params[i][3]
            A_i = self.dh_transform(theta, d, a, alpha)
            T = T @ A_i

            # COM position in world frame
            com_local = np.append(self.link_com_positions[i], 1)  # homogeneous
            com_world = T @ com_local
            p_com = com_world[0:3]

            # Now compute torque contribution from this link
            # Do partial FK again to get z_j and o_j for j <= i
            T_partial = np.eye(4)
            for j in range(i+1):
                theta_j = q[j] + self.dh_params[j][0]
                d_j     = self.dh_params[j][1]
                a_j     = self.dh_params[j][2]
                alpha_j = self.dh_params[j][3]
                A_j = self.dh_transform(theta_j, d_j, a_j, alpha_j)
                z = T_partial[0:3, 2]
                o = T_partial[0:3, 3]
                r = p_com - o
                G[j] += -self.link_masses[i] * self.gravity @ np.cross(z, r)
                T_partial = T_partial @ A_j

        return G

    def dynamics_step(self, q, q_dot, tau, dt):
        M = self.mass_matrix(q)
        G = self.gravity_torques(q)
        C = np.zeros((6, 6))  # placeholder

        q_ddot = np.linalg.solve(M, tau - C @ q_dot - G)

        q_dot_new = q_dot + q_ddot * dt
        q_new = q + q_dot_new * dt
        return q_new, q_dot_new


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
        return position, rotation, T

    def jacobian(self, q):
        T = np.eye(4)
        origins = [T[0:3, 3]]     # o_0
        z_axes = [T[0:3, 2]]      # z_0

        # Forward kinematics to collect all o_i and z_i
        for i in range(6):
            theta = q[i] + self.dh_params[i][0]
            d     = self.dh_params[i][1]
            a     = self.dh_params[i][2]
            alpha = self.dh_params[i][3]
            A_i = self.dh_transform(theta, d, a, alpha)
            T = T @ A_i

            origins.append(T[0:3, 3])     # o_i
            z_axes.append(T[0:3, 2])      # z_i

        o_n = origins[-1]
        J = np.zeros((6, 6))

        for i in range(6):
            z = z_axes[i]
            o_i = origins[i]
            J_v = np.cross(z, o_n - o_i)
            J_w = z
            J[0:3, i] = J_v
            J[3:6, i] = J_w

        return J

    # Compute rotation error as a 3D vector using quaternion error
    def orientation_error(self, R_current, q_goal):
        # Convert rotation matrix to quaternion
        r_current = R.from_matrix(R_current)
        q_current = r_current.as_quat()  # [x, y, z, w]

        # Normalize input quaternion (goal)
        q_goal = q_goal / np.linalg.norm(q_goal)
        r_goal = R.from_quat(q_goal)

        # Relative rotation: q_err = q_goal * q_current^-1
        r_err = r_goal * r_current.inv()
        angle_axis = r_err.as_rotvec()  # rotation vector

        return angle_axis  # Δθ: axis * angle

