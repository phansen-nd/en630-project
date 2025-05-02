import numpy as np

class UR5FrameConverter:
    def __init__(self):
        # Rotation from DH base frame to ROS URDF base_link frame
        self.R_align = np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [-1, 0, 0]
        ])

    def transform_position(self, p_dh: np.ndarray) -> np.ndarray:
        return self.R_align @ p_dh

    def transform_rotation(self, R_dh: np.ndarray) -> np.ndarray:
        return self.R_align @ R_dh

    def transform_jacobian(self, J_dh: np.ndarray) -> np.ndarray:
        J_ros = np.zeros_like(J_dh)
        J_ros[0:3, :] = self.R_align @ J_dh[0:3, :]   # Linear velocity part
        J_ros[3:6, :] = self.R_align @ J_dh[3:6, :]   # Angular velocity part
        return J_ros

    def transform_wrench(self, f_dh: np.ndarray) -> np.ndarray:
        return self.R_align @ f_dh

    def transform_inertia(self, I_dh: np.ndarray) -> np.ndarray:
        return self.R_align @ I_dh @ self.R_align.T
