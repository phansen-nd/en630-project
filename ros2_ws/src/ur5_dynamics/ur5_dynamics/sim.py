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

        current_position, _, _ = self.model.forward_kinematics(self.q)

        # todo: replace with PD controller that takes in desired joint angles
        desired_q_dot = np.array([0.5, 0.0, 0.0, 0.0, 1.0, 1.0])

        self.q_dot = desired_q_dot  # will add acceleration once we have dynamic model
        self.q += self.q_dot * self.dt

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
