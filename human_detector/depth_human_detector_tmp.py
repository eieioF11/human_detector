import os
import numpy as np
# ros2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import torch
from ultralytics import YOLO
from message_filters import SimpleFilter, Subscriber, Cache, TimeSynchronizer, ApproximateTimeSynchronizer

class HumanDetector(Node):
	# ノード名
	SELFNODE = "human_detector"
	def __init__(self):
		# ノードの初期化
		super().__init__(self.SELFNODE)
		self.get_logger().info("%s initializing..." % (self.SELFNODE))
		# モデルの読み込み
		model_name = self.param("model_name", 'yolov8x-seg.pt').string_value # yolov8n.pt, yolov8x-seg.pt ,yolov8x-pose.pt
		self.model = YOLO(model_name)
		# self.model.export(format="engine")
		# self.tensorrt_model = YOLO(model_name.replace('.pt', '.engine'))
		# ros2 init
		self.human_image_pub_ = self.create_publisher(Image,'human_detector/depth_human_image', 1)
		self.resaul_image_pub_ = self.create_publisher(Image,'human_detector/resaul_image', 1)
		self.image_sub_ = Subscriber(self,Image,'image_raw', qos_profile=ReliabilityPolicy.RELIABLE)
		self.depth_mage_sub_ = Subscriber(self,Image,'depth', qos_profile=ReliabilityPolicy.RELIABLE)
		self.image_sub_ = self.create_subscription(Image,'image_raw', self.image_callback, qos_profile=ReliabilityPolicy.RELIABLE)
		self.depth_sub_ = self.create_subscription(Image,'depth', self.depth_callback, qos_profile=ReliabilityPolicy.RELIABLE)
		self.timer_ = self.create_timer(0.001, self.timer_callback)
		# self.time_sync = TimeSynchronizer([self.image_sub_, self.depth_mage_sub_],100)
		# self.time_sync = ApproximateTimeSynchronizer([self.image_sub_, self.depth_mage_sub_],100,delay)
		# self.time_sync.registerCallback(self.sync_callback)
		self.bridge_ = CvBridge()
		self.image = None
		self.depth = None
		self.get_logger().info("end of %s initializing..." % (self.SELFNODE))

	def __del__(self):
		self.get_logger().info("%s done." % self.SELFNODE)


	def param(self, name, value):
		self.declare_parameter(name, value)
		return self.get_parameter(name).get_parameter_value()

	def image_callback(self, image):self.image = image
	def depth_callback(self, depth):self.depth = depth
	def timer_callback(self):
		if self.depth is not None and self.image is not None:
			self.sync_callback(self.image, self.depth)

	def sync_callback( self, image, depth):
		print("image_callback")
		raw_image = self.bridge_.imgmsg_to_cv2(image)
		d_img = self.bridge_.imgmsg_to_cv2(depth)
		img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
		# 人物検出
		# results = self.tensorrt_model.track(source=img, show=False, save=True)
		results = self.model.track(source=img, show=False, save=False)
		# get predict result
		human_depth_image = None
		human_mask = np.zeros(d_img.shape, dtype = np.uint16)
		for result in results:
			boxes = result.boxes  # Boxes object for bbox outputs
			masks = result.masks  # Masks object for segmentation masks outputs
			names = result.names
			speed = result.speed
			i=0
			# print(f"Number of boxes = {len(boxes)}")
			# print("-----")
			for box in boxes:
				cls = int(box.cls[0].item())
				i += 1
				if names[cls] != "person":
					continue
				# ids = box.id
				pos = box.xyxy[0]
				# conf = box.conf[0].item()
				# clsnm = names[cls]
				# keypoint = result.keypoints.xy[i]
				# print(f"+++++ Box-{i}")
				# print(f"class = {cls}:{clsnm}"); # 検知アイテムのクラス
				# print(f"conf = {conf:.5f}"); # 検知アイテムの信頼度
				# print(f"id = {ids}"); # 検知アイテムのID
				# # print(f"position = X0:{x0} Y0:{y0} X1:{x1} Y1:{y1}"); # 検知アイテムのPosition
				# print("+++++")
				if masks is not None:
					#  Extract contour result
					contour = masks.xy[i-1].astype(np.int32).reshape(-1, 1, 2)
					mask = np.zeros(d_img.shape, dtype = np.uint16)
					cv2.drawContours(mask, [contour], -1, (65536, 65536, 65536), cv2.FILLED)
					human_mask = cv2.bitwise_or(human_mask, mask)
				else:
					x0 = int(pos[0].item())
					y0 = int(pos[1].item())
					x1 = int(pos[2].item())
					y1 = int(pos[3].item())
					cv2.rectangle(human_mask, (x0, y0), (x1, y1), (65536, 65536, 65536), -1)
				kc = 0
			# print("names:",names)
			# print ("masks:",masks)
			# print ("speed:",speed)

			# Display the annotated frame
		if len(results[0]) > 0:
			results_frame = results[0].plot(line_width=5, font_size=1)
			human_depth_image = cv2.bitwise_and(d_img, human_mask)

		img_msg = self.bridge_.cv2_to_imgmsg(results_frame, encoding='bgr8')
		img_msg.header = image.header
		self.resaul_image_pub_.publish(img_msg)
		if human_depth_image is not None:
			img_msg = self.bridge_.cv2_to_imgmsg(human_depth_image, encoding='16UC1')
			img_msg.header = image.header
			self.human_image_pub_.publish(img_msg)

def main(args=None):
	try:
		rclpy.init(args=args)
		node=HumanDetector()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		# 終了処理
		rclpy.shutdown()


if __name__ == '__main__':
	main()