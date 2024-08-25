#!/usr/bin/env python

import rospy
import actionlib
import tf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
import math as m


# Some environment dependent variables
object_diameter=0.05
desired_dist_to_obj = 0.4
image_width = 1920
image_heights = 1080
vertical_FOV_degrees = 40
vertical_FOV_rad = np.deg2rad(vertical_FOV_degrees)
robot_namespace = "robot1"
depth_img_topic = "/camera/depth/image_raw"
camera1_topic = "/camera/rgb/image_raw"
camera2_topic = "/camera/rgb/image_raw"
cmd_vel_topic = "/cmd_vel"
gripper_topic = "/gripper/command"
signal_topic = "/object_signal"
move_base_topic = "/move_base"


class BallFollower:
    def __init__(self):
        rospy.init_node('ball_follower')

        # Move base client
        self.move_base_client = actionlib.SimpleActionClient(robot_namespace + move_base_topic, MoveBaseAction)
        self.move_base_client.wait_for_server()

        # Subscribers and publishers
        self.image_sub = rospy.Subscriber(robot_namespace + camera1_topic, Image, self.callback_camera0)
        #self.image_sub2 = rospy.Subscriber(robot_namespace + camera2_topic, Image, self.callback_camera1)
        self.image_dist_sub = rospy.Subscriber(robot_namespace + depth_img_topic, Image, self.callback_depth)
        self.pub = rospy.Publisher(robot_namespace + cmd_vel_topic, Twist, queue_size=10)
        self.gripper_pub = rospy.Publisher(gripper_topic, String, queue_size=10)
        self.stacking_signal_pub = rospy.Publisher(signal_topic, String, queue_size=10)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize TF listener
        self.tf_listener = tf.TransformListener()

        # Variables
        self.ball_detected = False
        self.ball_position = None
        self.ball_position_grasping = None
        self.distance_to_ball = None
        self.ball_color = ""

        # Define waypoints
        self.waypoints = [
            (-0.75, 2.5, 1.0, 0.0),
            (-3.88, 2.3, -0.707, 0.707),
            (-3.9, -1.85, 0.0, 1.0),
            (-0.85, -1.76, 0.707, 0.707),
            (-0.5, 0.0, 1.0, 0.0)
        ]
        self.current_waypoint_index = 0
        self.state = 'PATROLLING'

        self.run()

        return



    

    def callback_depth(self, img):
        # Compute the depth of the stored center of mass of the object, if one is detected.
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
            if self.ball_detected and self.ball_position:
                depth_value = cv_image[int(self.ball_position[1]), int(self.ball_position[0])]
                if not np.isnan(depth_value) and depth_value > 0:
                    self.distance_to_ball = depth_value
                else:
                    self.distance_to_ball = None
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        return


    def callback_camera0(self, data):
        # Detect objects and save the center of mass of the largest red, green or blue object
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        cv_image = self.detect_object(cv_image, 0)
        cv2.imshow("Image Window", cv_image)
        cv2.waitKey(3)

        return


    def callback_camera1(self, data):
        # Just for the real robot, because it has more cameras. Useless for simulation
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        cv_image = self.detect_object(cv_image, 1)
        cv2.imshow("Image Window", cv_image)
        cv2.waitKey(3)

        return



    # Detects the largest object in the image and saves its center of mass in self.ballposition
    # The idea behind camera number is, that the secound camera of the real robot has to be used if the object is to close
    # This is because the first camera is to high and would look over the ball in f.e. 10cm distance, so the grasping coordinates have to be extracted from the second camera 
    def detect_object(self, image, camera_number):
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Parameters for the masks
        masks = {
            "red": ((0, 80, 80), (30, 500, 500)),
            "blue": ((180, 80, 80), (260, 500, 500)),
            "green": ((30, 80, 80), (180, 500, 500))
        }

        # Find largest red green and blue object in the image
        contours_rgb = []
        for color, (low, high) in masks.items():
            mask_frame = cv2.inRange(hsv_frame, low, high)
            contours, _ = cv2.findContours(mask_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image, contour = self.label_and_draw_largest_contour(contours, image, color)
            if len(contour) > 0:
                contours_rgb.append((contour, color))

        # If r,g or b contours were detected, save position of largest
        if contours_rgb:
            contour_max, color = max(contours_rgb, key=lambda x: cv2.contourArea(x[0]))
            self.ball_detected = True
            M = cv2.moments(contour_max)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if not cx:
                    print("ERROR")
                if camera_number == 0:
                    self.ball_position = (cx, cy)
                    self.ball_color = color
                elif camera_number == 1:
                    self.ball_position_grasping = (cx, cy)
            else:
                self.ball_detected = False
        else:
            self.ball_detected = False

        return image


    # Uses the image and the found contours to detect the max contour and check what object it is and to draw it in the image
    def label_and_draw_largest_contour(self, contours, image, color_text):
        if contours:
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            area = cv2.contourArea(contour)
            if cv2.arcLength(contour, True) ** 2 > 0:
                circularity = (4 * np.pi * area) / (cv2.arcLength(contour, True) ** 2)
                if circularity > 0.85:
                    shape = "Ball" + " " + color_text
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                    cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    if len(approx) < 7 and len(approx) >= 4 and cv2.isContourConvex(approx):
                        shape = "Cube" + " " + color_text
                        cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
                        cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                contour = []
        else:
            contour = []

        return image, contour



    # Transforms coordinates from source to dest
    def transform(self, x, y, z,source,dest):
        point = PointStamped()
        point.header.frame_id = source
        point.header.stamp = rospy.Time(0)
        point.point.x = x
        point.point.y = y
        point.point.z = z
        trans_point = self.tf_listener.transformPoint(dest, point)

        return trans_point

    # Transforms coordinates from source to dest and returns the robots orientation
    def transform_and_get_orientation(self, x, y, z,source,dest):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/robot1_tf/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("TF Exception: %s" % e)
            return None, None
        
        _, _, orientation_z, orientation_w = rot
        rospy.loginfo(trans)
        
        point = PointStamped()
        point.header.frame_id = source
        point.header.stamp = rospy.Time(0)
        point.point.x = x
        point.point.y = y
        point.point.z = z
        trans_point = self.tf_listener.transformPoint(dest, point)

        return trans_point, orientation_z, orientation_w



    # Just moves to the position x,y,z,w, but if the robot is patrolling and it detects an object, it should approach the object
    def move_to(self, x, y, z, w):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = z
        goal.target_pose.pose.orientation.w = w
        
        self.move_base_client.send_goal(goal)

        # Only if the robot is patrolling, it should stop if it detects an object
        self.current_goal = goal
        if self.state == "PATROLLING":
            while not self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                if self.ball_detected:
                    self.move_base_client.cancel_goal()
                    break
                if self.move_base_client.get_state() in [actionlib.GoalStatus.PREEMPTED, actionlib.GoalStatus.RECALLED]:
                    rospy.loginfo("Goal was preempted or recalled.")
                    break
                rospy.sleep(0.1)                                                                                            # Check 10 times a secound
        else:
            self.move_base_client.wait_for_result()
        
        return



    def orient_to_object(self, const_speed, confidence):
        msg = Twist()
        counter = 0

        # Rotate for alignment
        rate = rospy.Rate(120)
        while True:
            obj_x = self.ball_position[0] - (image_width/2)                 # How far is object from image middle, Camera 0 is used because this camera is for navigation, other for grasping
            speed = abs((0.1 * obj_x / (image_width/2))) + const_speed      # Scale rotation speed with distance to image mid

            if abs(obj_x)<3:                                                # Close enough to image middle
                msg.angular.z = 0
                msg.linear.x = 0
                self.pub.publish(msg)
                counter = counter + 1
                if counter >= confidence:
                    break

            elif obj_x >= 3:                                                # Turn left
                msg.angular.z = -speed
                msg.linear.x = 0

            elif obj_x <= -3:                                               # Turn right
                msg.angular.z = speed
                msg.linear.x = 0

            self.pub.publish(msg)
            rate.sleep()

        return
    

    # First aling the camera with the detected object, then compute the coordinates of the ball and the coordinates of the point with the desired_distance and move to it
    def approach_object(self, desired_distance):     
        # Orient to ball with speed at least 0.016 and the object beeing in the center at least 20 times
        self.orient_to_object(0.018,20)                            


        depth = self.distance_to_ball + (object_diameter/2)                     # Adjust for thickness of object

        y_obj = 0
        x_obj = 0
        z_obj = depth                                                           # Account for object diameter and distance for grasping, the camera is centered with the object

        map_point, z, w = self.transform_and_get_orientation(x_obj,y_obj,z_obj, "/robot1_tf/camera_depth_optical_frame", "/map")
        rospy.loginfo("Coordinates of ball in map coordinate system: ({}, {}, {})".format(map_point.point.x, map_point.point.y, map_point.point.z))


        z_robot_dest = depth - (object_diameter / 2) - desired_distance
        y_robot_dest = 0
        x_robot_dest = 0

        # Transform the point to the map coordinate system
        map_point, z, w = self.transform_and_get_orientation(x_robot_dest,y_robot_dest,z_robot_dest, "/robot1_tf/camera_depth_optical_frame", "/map")

        rospy.loginfo("Coordinates to move to in map coordinate system: ({}, {}, {})".format(map_point.point.x, map_point.point.y, map_point.point.z))
        
        # Dont approach if on the wrong side of the map
        if map_point.point.x>0:
            rospy.loginfo("Saving from wrong object")
            self.state = "SAVE"
            return
        
        self.move_to(map_point.point.x,map_point.point.y,z,w)   # Move to the map point with same orientation the robot has
        return


    def orient_slow_and_get_obj_coords(self):
        # Orient to ball with speed at least 0.012 and the object beeing in the center at least 20 times
        self.orient_to_object(0.014,20)
        
        #vertical_position_of_ball = self.ball_position[1]
        #theta_v = (vertical_position_of_ball / (image_heights/2)) * vertical_FOV_rad
        
        
        #x_pos = -(self.distance_to_ball + object_diameter/2) * np.sin(theta_v)
        #y_pos = 0.0
        #z_pos = (self.distance_to_ball + object_diameter/2) * np.cos(theta_v)               # Account for object diameter, the camera is centered with the object

        
        
        #x_pos = 0
        #y_pos = (vertical_position_of_ball - 540) * self.distance_to_ball / 525
        #z_pos = self.distance_to_ball + object_diameter/2                                   # Account for object diameter, the camera is centered with the object

        #x_pos = 0 #self.distance_to_ball + object_diameter/2
        #y_pos = 0
        #z_pos = self.distance_to_ball + object_diameter/2


        alpha = m.asin(0.075/(self.distance_to_ball + object_diameter/2))
        x_pos = 0
        y_pos = m.sin(alpha) * (self.distance_to_ball + object_diameter/2)
        z_pos = m.cos(alpha) * (self.distance_to_ball + object_diameter/2)
        rospy.loginfo(f"x: {x_pos}, y:{y_pos}, z:{z_pos}")

        

        map_point, z , w = self.transform_and_get_orientation(x_pos, y_pos, z_pos, "robot1_tf/camera_depth_optical_frame", "/map")
        #rospy.loginfo(f"Map point initial: {map_point.point.x} y: {map_point.point.y}")
        #rospy.loginfo(f"Difference x:{(m.asin(z)/ (m.pi/2) * -0.141 - m.asin(w) / (m.pi/2) * 0.094)} y: {(m.asin(w) / (m.pi/2) * 0.0586 - m.asin(z) / (m.pi/2) * 0.04) }")
        #map_point.point.x = map_point.point.x + (abs(m.asin(z)/ (m.pi/2)) * -0.141 - abs(m.asin(w) / (m.pi/2)) * 0.094) 
        #map_point.point.y = map_point.point.y + (m.asin(w) / (m.pi/2) * -0.2 - m.asin(z) / (m.pi/2) * 0.12) 
        return map_point


    def handle_object(self):
        rospy.sleep(0.2)

        if self.ball_detected and self.distance_to_ball is not None:
            self.approach_object(desired_dist_to_obj)
            rospy.loginfo("Fine tune orientation")

            grasp_position = self.orient_slow_and_get_obj_coords()
            rospy.loginfo(f"Map point of Ball: x: {grasp_position.point.x}, y: {grasp_position.point.y}, z: {grasp_position.point.z}")

            # Sleep a bit so that i can have a look at the situation
            rospy.sleep(2)

            # If the state is save, an object was detected, that is on the wrong side of the map, then we do not want to move to finish, but to the next waypoint
            if self.state != "SAVE":
                self.state = 'MOVE_TO_STACKING_STATION'
        return



    def run(self):
        # Main loop of the program
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # MOVE_TO_STACKING_STATION closes the gripper, lets the robot move to the stacking station, opens the gripper, send message to other robot with color information, and moves back to first waypoint in this function
            # This is to avoid the robot from following balls in the wrong side of the map from the stacking station which would cancel the way to first waypoint if state was PATROLLING
            if self.state == 'MOVE_TO_STACKING_STATION':
                # Move to stacking station 
                rospy.loginfo("Moving to finish position")
                self.close_gripper()
                self.move_to(1.0, 0.0, 0.0, 1.0)
                self.open_gripper()
                self.stacking_signal_pub.publish(self.ball_color)
                self.current_waypoint_index = 0
                self.move_to(*self.waypoints[self.current_waypoint_index])           # Move to first position without accounting for new objects on wrong side
                self.state = 'PATROLLING'
            

            # If an object was detected on the wrong side, move to the next waypoint and ignore all object on its way to it
            elif self.state == 'SAVE':
                self.move_to(*self.waypoints[self.current_waypoint_index]) 
                self.state = 'PATROLLING'

            # If a ball is detected bring it to stacking station
            elif self.ball_detected or self.state == "BALL_DETECTED":
                # Approach the Object
                rospy.loginfo("Ball detected! Moving towards it.")
                self.state = "BALL_DETECTED"
                self.handle_object()

            # Move between waypoints, if object is detected handle the object
            elif self.state == 'PATROLLING':
                # Move to the next waypoint
                waypoint = self.waypoints[self.current_waypoint_index]
                rospy.loginfo(f"Moving to waypoint: {waypoint}")
                self.move_to(*waypoint)
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)

            rate.sleep()




    def close_gripper(self):
        # Only closes the gripper should first move to the object position but this is not implemented yet
        rospy.loginfo("Close gripper")
        self.gripper_pub.publish("close")


    def open_gripper(self):
        rospy.loginfo("Open gripper")
        self.gripper_pub.publish("open")


if __name__ == '__main__':
    try:
        BallFollower()
    except rospy.ROSInterruptException:
        pass
