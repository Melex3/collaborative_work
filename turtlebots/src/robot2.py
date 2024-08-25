#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from std_msgs.msg import String

class BallFollower:
    def __init__(self):
        robot_namespace = "robot2"
        rospy.init_node('ball_collector')
        

        # Move base used to send positions the robot moves to
        self.move_base_client = actionlib.SimpleActionClient(robot_namespace + '/move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()

        # Subscriber to the robot's camera and depth camera, vel publisher and gripper publisher
        self.gripper_pub = rospy.Publisher('/gripper/command', String, queue_size=10)
        self.stacking_signal_sub = rospy.Subscriber("/object_signal", String, self.stacking_signal_callback)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Allow up to one second to connection
        rospy.sleep(1)

        
        # Define drop-off points for each color
        self.drop_off_points = {
            "blue": (4.3, 2.0, 0.0, 1.0),
            "green": (4.3, 0.0, 0.0, 1.0),
            "red": (4.3, -2.0, 0.0, 1.0)
        }
        
        # Define states
        #self.state = 'MOVING_TO_INITIAL'
        
        # Run the main loop
        self.run()


    def stacking_signal_callback(self,msg):
        rospy.loginfo(msg)
        self.move_to(1.0, -0.2, 0.707, 0.707)
        self.close_gripper()
        drop_off_point = self.drop_off_points[msg.data]
        self.move_to(*drop_off_point)
        self.open_gripper()
        return
    
    def open_gripper(self):
        self.gripper_pub.publish("open")
        rospy.loginfo("Gripper opened")

    def close_gripper(self):
        self.gripper_pub.publish("close")
        rospy.loginfo("Gripper closed")


    # Simply moves to x,y coordinate with angle z and w
    def move_to(self, x, y, z, w):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = z
        goal.target_pose.pose.orientation.w = w
        
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()


    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        BallFollower()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
