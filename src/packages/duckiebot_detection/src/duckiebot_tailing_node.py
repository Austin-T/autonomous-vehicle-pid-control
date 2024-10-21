#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, BoolStamped, VehicleCorners, LEDPattern
from duckietown_msgs.srv import ChangePattern, SetCustomLEDPattern
from dt_apriltags import Detector, Detection
import yaml

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]  # for yellow mask
DEBUG = False
ENGLISH = False


class DuckiebotTailingNode(DTROS):

    def __init__(self, node_name):
        super(DuckiebotTailingNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Publishers
        self.pub_mask = rospy.Publisher(
            "/" + self.veh + "/output/image/mask/compressed", CompressedImage, queue_size=1)
        self.vel_pub = rospy.Publisher(
            "/" + self.veh + "/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_img_bool = True

        # Subscribers
        self.sub_camera = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                           CompressedImage, self.img_callback, queue_size=1, buff_size="20MB")
        self.sub_distance = rospy.Subscriber(
            f'/{self.veh}/duckiebot_distance_node/distance', Float32, self.dist_callback)
        self.sub_detection = rospy.Subscriber(
            f'/{self.veh}/duckiebot_detection_node/detection', BoolStamped, self.detection_callback)
        self.sub_centers = rospy.Subscriber(
            f'/{self.veh}/duckiebot_detection_node/centers', VehicleCorners, self.centers_callback, queue_size=1)

        # image processing tools
        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()

        # info from subscribers
        self.detection = False
        self.intersection_detected = False
        self.centers = None

        # find the calibration parameters
        # for detecting apriltags
        camera_intrinsic_dict = self.readYamlFile(
            f'/data/config/calibrations/camera_intrinsic/{self.veh}.yaml')

        self.K = np.array(
            camera_intrinsic_dict["camera_matrix"]["data"]).reshape((3, 3))
        self.R = np.array(
            camera_intrinsic_dict["rectification_matrix"]["data"]).reshape((3, 3))
        self.DC = np.array(
            camera_intrinsic_dict["distortion_coefficients"]["data"])
        self.P = np.array(
            camera_intrinsic_dict["projection_matrix"]["data"]).reshape((3, 4))
        self.h = camera_intrinsic_dict["image_height"]
        self.w = camera_intrinsic_dict["image_width"]

        f_x = camera_intrinsic_dict['camera_matrix']['data'][0]
        f_y = camera_intrinsic_dict['camera_matrix']['data'][4]
        c_x = camera_intrinsic_dict['camera_matrix']['data'][2]
        c_y = camera_intrinsic_dict['camera_matrix']['data'][5]
        self.camera_params = [f_x, f_y, c_x, c_y]

        # initialize apriltag detector
        self.at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)
        self.at_distance = 0
        self.intersections = {
            133: 'INTER',  # T intersection
            153: 'INTER',  # T intersection
            62: 'INTER',  # T intersection
            58: 'INTER',  # T intersection
            162: 'STOP',  # Stop sign
            169: 'STOP'   # Stop sign
        }
        self.led_colors = {
            0: 'WHITE',
            1: 'RED'
        }

        # apriltag detection filters
        self.decision_threshold = 10
        self.z_threshold = 0.35

        # PID Variables for driving
        self.proportional = None
        if ENGLISH:
            self.offset = -170
        else:
            self.offset = 170
        self.varying_velocity = 0.25
        self.velocity = 0.25
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.025
        self.D = -0.0035
        self.last_error = 0
        self.last_time = rospy.get_time()
        self.calibration = 0.5

        # PID Variables for tailing a duckiebot
        self.distance = 0
        self.forward_P = 0.005
        self.forward_D = 0.001
        self.target = 0.40
        self.forward_error = 0
        self.last_fwd_err = 0
        self.last_distance = 0
        self.dist_margin = 0.05
        self.tailing = False

        self.camera_center = 340
        self.leader_x = self.camera_center
        self.straight_threshold = 75

        self.turn_speed = 0.15

        # Service proxies
        # rospy.wait_for_service(f'/{self.veh}/led_emitter_node/set_pattern')
        # self.led_service = rospy.ServiceProxy(f'/{self.veh}/led_emitter_node/set_pattern', ChangePattern)
        rospy.wait_for_service(
            f'/{self.veh}/led_emitter_node/set_custom_pattern')
        self.led_service = rospy.ServiceProxy(
            f'/{self.veh}/led_emitter_node/set_custom_pattern', SetCustomLEDPattern, persistent=True)

        self.loginfo("Initialized")
        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def run(self):
        if self.intersection_detected:
            self.intersection_sequence()
        elif self.detection:
            self.tailing = True
            self.tailPID()
        else:
            self.tailing = False
            self.drive()

    def dist_callback(self, msg):
        self.distance = msg.data

    def detection_callback(self, msg):
        self.detection = msg.data

    def centers_callback(self, msg):
        self.centers = msg.corners

        if self.detection and len(self.centers) > 0:
            # find the middle of the circle grid
            middle = 0
            i = 0
            while i < len(self.centers):
                middle += self.centers[i].x
                i += 1
            middle = middle / i

            # update the last known position of the bot ahead
            self.leader_x = middle

    def img_callback(self, msg):
        img = self.jpeg.decode(msg.data)
        self.intersection_detected = self.detect_intersection(msg)
        crop = img[:, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None

        # debugging
        if self.pub_img_bool:
            rect_img_msg = CompressedImage(
                format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)

    def old_left_turn(self):
        rospy.loginfo("Beginning left turn")

        self.set_lights("left")

        self.twist.v = self.turn_speed
        self.twist.omega = 3

        start_time = rospy.get_time()
        while rospy.get_time() < start_time + 0.8:

            self.vel_pub.publish(self.twist)

    def old_right_turn(self):
        rospy.loginfo("Beginning right turn")

        self.set_lights("right")

        self.twist.v = self.turn_speed
        self.twist.omega = -4

        start_time = rospy.get_time()
        while rospy.get_time() < start_time + 0.6:

            self.vel_pub.publish(self.twist)

    def tailPID(self):
        # forward error is negative if duckiebot is too close, positive if too far
        self.forward_error = self.distance - self.target
        tail_P = self.forward_error * self.forward_P

        tail_d_error = (self.forward_error - self.last_fwd_err) / \
            (rospy.get_time() - self.last_time)
        self.last_fwd_err = self.forward_error
        self.last_time = rospy.get_time()
        tail_D = -tail_d_error * self.forward_D

        if self.forward_error < 0:
            self.twist.v = 0
            self.twist.omega = 0
        else:
            self.twist.v = tail_P + tail_D
            if self.proportional is None:
                self.twist.omega = 0
            else:
                # P Term
                P = -self.proportional * self.P

                # D Term
                d_error = (self.proportional - self.last_error) / \
                    (rospy.get_time() - self.last_time)
                self.last_error = self.proportional
                self.last_time = rospy.get_time()
                D = d_error * self.D
                self.twist.omega = P + D

        self.last_distance = self.distance
        self.vel_pub.publish(self.twist)

    def drive(self):
        if self.proportional is None:
            self.twist.omega = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / \
                (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            self.twist.v = self.velocity
            self.twist.omega = P + D

        self.vel_pub.publish(self.twist)

    def intersection_sequence(self):
        # for now
        rospy.loginfo("INTERSECTION DETECTED. at {} INTERSECTION DETECTED. INTERSECTION DETECTED.".format(str(self.at_distance)))

        if self.check_for_leader():
            # stopping_threshold = 0.1
            stopping_threshold = 0.3
            while self.twist.v > 0 and self.forward_error > stopping_threshold:
                self.tailPID()
            self.stop()
            # wait until the leading duckiebot has moved <- if self.distance isn't updating properly, this won't work
            starting_distance = self.distance
            movement_threshold = 0.3
            while self.detection and self.distance < starting_distance + movement_threshold:
                # once in a while there's an inaccurate reading and that's going to set this off
                # so have it wait so it doesn't start moving immediately
                self.pass_time(2)
                rospy.loginfo(self.distance)

        # latency between detecting intersection and stopping
        self.pub_straight(linear=0)
        wait_time = self.at_distance * 2  # 1.5 # seconds
        self.pass_time(wait_time)

        self.stop()
        self.pass_time(6)
        # move forward a bit
        self.pub_straight(linear=0)
        self.pass_time(2)
        self.stop()
        if self.detection:
            # drive straight through intersection
            self.set_lights("off")
            return

        self.left_turn()
        self.stop()
        if self.detection:
            self.set_lights("off")
            return

        # correction
        self.right_turn()
        self.stop()
        # turn right
        self.right_turn()
        self.stop()
        if self.detection:
            self.set_lights("off")
            return

        # correction
        self.left_turn()
        self.left_turn()
        self.stop()
        self.proportional = None
        self.set_lights("off")

    def right_turn(self):
        self.set_lights("right")
        self.twist.v = 0
        self.twist.omega = -12
        self.vel_pub.publish(self.twist)
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + 0.6:
            continue

    def left_turn(self):
        self.set_lights("left")
        self.twist.v = 0
        self.twist.omega = 12
        self.vel_pub.publish(self.twist)
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + 0.6:
            continue

    def stop(self):
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        self.set_lights("stop")

    def pub_straight(self, linear=None):
        self.twist.v = self.velocity
        if linear is not None:
            self.twist.omega = linear
        else:
            self.twist.omega = self.calibration
        self.vel_pub.publish(self.twist)

    def pass_time(self, t):
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + t:
            continue

    def check_for_leader(self):
        # sometimes we get one-off readings. This helps to make sure there's a duckiebot.
        for i in range(8):
            if self.detection:
                return True
        return False

    def tailing_intersection(self):
        # for now
        rospy.loginfo("DUCKIEBOT INTERSECTION DETECTED. at {} DUCKIEBOT INTERSECTION DETECTED. DUCKIEBOT INTERSECTION DETECTED.".format(str(self.at_distance)))

        # stop behind the leading duckiebot
        # 10 cm, can change to 0.05 m. Chose 10 bc bot underestimates distance
        stopping_threshold = 0.1
        while self.forward_error > stopping_threshold:
            self.tailPID()
        # stop
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)

        # wait until the leading duckiebot has moved
        starting_distance = self.distance
        movement_threshold = 0.15
        while self.detection and self.distance < starting_distance + movement_threshold:
            # once in a while there's an inaccurate reading and that's going to set this off
            # so have it wait so it doesn't start moving immediately
            wait_time = 2  # seconds
            start_time = rospy.get_time()
            while rospy.get_time() < start_time + wait_time:
                continue

        # move forward a bit
        self.twist.v = self.velocity
        self.twist.omega = self.calibration
        self.vel_pub.publish(self.twist)
        wait_time = 0.5  # seconds
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + wait_time:
            continue
        # stop at red line
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        self.set_lights("stop")

        wait_time = 5  # seconds
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + wait_time:
            continue

        # move forward a bit
        self.twist.v = self.velocity
        self.twist.omega = self.calibration
        self.vel_pub.publish(self.twist)
        wait_time = 1  # seconds
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + wait_time:
            continue

        # stop
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)

        if self.detection:
            # drive straight through intersection
            wait_time = 0.75  # seconds
            start_time = rospy.get_time()
            '''
            while rospy.get_time() < start_time + wait_time:
                continue
            '''
            self.set_lights("off")
            self.tailPID()
        else:
            # turn left
            self.twist.v = 0
            self.twist.omega = 4
            self.vel_pub.publish(self.twist)
            start_time = rospy.get_time()
            while rospy.get_time() < start_time + 0.6:
                continue

            self.twist.v = 0
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)

            start_time = rospy.get_time()
            while rospy.get_time() < start_time + 0.6:
                continue

            if self.detection:
                self.set_lights("off")
                self.tailPID()
            else:
                # turn right
                self.twist.v = 0
                self.twist.omega = -4
                self.vel_pub.publish(self.twist)
                start_time = rospy.get_time()
                while rospy.get_time() < start_time + 1.2:
                    continue

                self.twist.v = 0
                self.twist.omega = 0
                self.vel_pub.publish(self.twist)

                start_time = rospy.get_time()
                while rospy.get_time() < start_time + 0.6:
                    continue

                if self.detection:
                    self.set_lights("off")
                    self.tailPID()

    def detect_intersection(self, img_msg):
        # detect an intersection by finding the corresponding apriltags
        cv_image = None
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        except CvBridgeError as e:
            self.log(e)
            return []

        # undistort the image
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.DC, (self.w, self.h), 0, (self.w, self.h))
        image_np = cv2.undistort(cv_image, self.K, self.DC, None, newcameramtx)

        # convert the image to black and white
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # detect tags present in image
        tags = self.at_detector.detect(
            image_gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=0.065)

        closest_tag_z = 1000
        closest = None

        for tag in tags:
            # ignore distant tags and tags with bad decoding
            z = tag.pose_t[2][0]
            if tag.decision_margin < self.decision_threshold or z > self.z_threshold:
                continue

            # update the closest-detected tag if needed
            if z < closest_tag_z:
                closest_tag_z = z
                closest = tag

        if closest:
            if closest.tag_id in self.intersections:
                self.at_distance = closest.pose_t[2][0]
                return True
        return False

    def set_lights(self, state):

        msg = LEDPattern()
        if state == "left":
            msg.color_list = ['yellow','yellow','switchedoff','switchedoff','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 2
            msg.frequency_mask = [1, 1, 0, 0, 0]
        elif state == "right":
            msg.color_list = ['switchedoff','switchedoff','yellow','yellow','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 2
            msg.frequency_mask = [0, 0, 1, 1, 0]
        elif state == "stop":
            msg.color_list = ['switchedoff','red','switchedoff','red','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 0
            msg.frequency_mask = [0, 0, 0, 0, 0]
        elif state == "off":
            msg.color_list = ['switchedoff','switchedoff','switchedoff','switchedoff','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 0
            msg.frequency_mask = [0, 0, 0, 0, 0]

        self.led_service(msg)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)

    def readYamlFile(self, fname):
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         % (fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


if __name__ == "__main__":
    node = DuckiebotTailingNode("duckiebot_tailing_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.run()
        rate.sleep()
