import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from ur5_dynamics.ur5_model import UR5Model
from scipy.spatial.transform import Rotation as R

import numpy as np

scripted_poses = [
    ([0.0, 0.3, 0.4], R.from_euler('xyz', [0, 0, 0]).as_quat()),
    ([0.4, 0.3, 0.0], R.from_euler('xyz', [0, np.pi/2, 0]).as_quat()),
    ([0.5, 0.0, 0.4], R.from_euler('xyz', [np.pi/2, np.pi, 0]).as_quat()),
    ([0.5, 0.3, 0.0], R.from_euler('xyz', [np.pi/2, 0, 0]).as_quat()),
    ([0.6, 0.0, 0.4], R.from_euler('xyz', [np.pi, 0, 0]).as_quat()),
    ([0.6, 0.3, 0.0], R.from_euler('xyz', [np.pi, np.pi/2, 0]).as_quat()),
]

class DynamicsSimulator(Node):
    def __init__(self, scripted_poses=scripted_poses):
        super().__init__('sim')
        self.get_logger().info("Started sim node...")

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/goal_pose', self.pose_callback, 10)

        self.scripted_poses = scripted_poses
        self.current_pose_index = 0
        self.goal_pose = self.make_pose_msg(*self.scripted_poses[0])

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

        current_position, current_rotation, _ = self.model.forward_kinematics(self.q)
        goal_position = np.array([self.goal_pose.pose.position.x, self.goal_pose.pose.position.y, self.goal_pose.pose.position.z])
        goal_rotation = np.array([self.goal_pose.pose.orientation.x, self.goal_pose.pose.orientation.y, self.goal_pose.pose.orientation.z, self.goal_pose.pose.orientation.w])
        
        q_solution = self.model.inverse_kinematics(self.q, goal_position, goal_rotation)

        alpha = 0.1
        self.q = self.q + alpha * (q_solution - self.q)

        self.publish_position()

        pos_err = np.linalg.norm(goal_position - current_position)
        rot_err = np.linalg.norm(self.model.orientation_error(current_rotation, goal_rotation))

        # If close enough, advance to the next pose in the script
        if pos_err < 0.002 and rot_err < 0.1:
            self.current_pose_index += 1
            if self.current_pose_index < len(self.scripted_poses):
                next_pos, next_ori = self.scripted_poses[self.current_pose_index]
                self.goal_pose = self.make_pose_msg(next_pos, next_ori)
                self.get_logger().info(f"Moving to scripted pose {self.current_pose_index + 1}")
            else:
                self.get_logger().info("Completed all scripted poses.")

    def publish_position(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.q.tolist()
        self.joint_pub.publish(msg)

    def pose_callback(self, msg):
        self.get_logger().info('Received goal pose')
        self.goal_pose = msg

    def make_pose_msg(self, position, orientation):
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = orientation
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = DynamicsSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
