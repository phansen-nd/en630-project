import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from ur5_dynamics.ur5_model import UR5Model

import numpy as np

class DynamicsSimulator(Node):
    def __init__(self):
        super().__init__('sim')
        self.get_logger().info("Started sim node...")

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/goal_pose', self.pose_callback, 10)

        self.goal_pose = None
        self.model = UR5Model()

        # From official UR5 urdf
        self.joint_names = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]

        # Simulation params
        self.dt = 0.05  # 50ms timestep (20Hz)
        self.q = np.zeros(6)      # Joint angles (rad)
        self.q_dot = np.zeros(6)  # Joint velocities (rad/s)

        self.timer = self.create_timer(self.dt, self.simulation_step)

    def simulation_step(self):
        if self.goal_pose is None:
            self.publish_position()
            return

        # Parse goal pose
        goal_position = np.array([self.goal_pose.pose.position.x, self.goal_pose.pose.position.y, self.goal_pose.pose.position.z])
        goal_rotation = np.array([self.goal_pose.pose.orientation.x, self.goal_pose.pose.orientation.y, self.goal_pose.pose.orientation.z, self.goal_pose.pose.orientation.w])
        current_position, current_rotation, _ = self.model.forward_kinematics(self.q)

        self.get_logger().info(f'Current position: {current_position}')
        self.get_logger().info(f'Target position: {goal_position}')

        e_p = goal_position - current_position
        e_R = self.model.orientation_error(current_rotation, goal_rotation)
        e = np.concatenate([e_p, e_R])

        Kp = np.diag([50.0]*3 + [10.0]*3)  # positional and rotational gains
        Kd = np.diag([10.0]*3 + [2.0]*3)   # damping gains

        # Optional: you can compute end-effector velocity using J * q_dot
        v_ee = self.model.jacobian(self.q) @ self.q_dot  # (6,)
        F_task = Kp @ e - Kd @ v_ee

        J = self.model.jacobian(self.q)
        tau = J.T @ F_task

        self.q, self.q_dot = self.model.dynamics_step(self.q, self.q_dot, tau, self.dt)

        self.publish_position()

    def publish_position(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.q.tolist()
        self.joint_pub.publish(msg)

    def pose_callback(self, msg):
        self.get_logger().info('Received goal pose')
        self.goal_pose = msg

def main(args=None):
    rclpy.init(args=args)
    node = DynamicsSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
