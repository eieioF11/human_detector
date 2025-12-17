import os
import numpy as np
# ros2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import torch

class TDMPCSocialNavigation(Node):
	# ノード名
	SELFNODE = "tdmpc_social_navigation"
	def __init__(self):
		# ノードの初期化
		super().__init__(self.SELFNODE)
		self.get_logger().info("%s initializing..." % (self.SELFNODE))
		# モデルの読み込み
		# model_name = self.param("model_name", 'yolov8x-pose.pt').string_value # yolov8n.pt,yolov8x-seg.pt
		# ros2 init
		# self.human_image_pub_ = self.create_publisher(Image,'tdmpc_social_navigation/human_image', 1)
		# self.image_sub_ = self.create_subscription(Image,'image_raw', self.image_callback, qos_profile=ReliabilityPolicy.RELIABLE)

	def __del__(self):
		self.get_logger().info("%s done." % self.SELFNODE)


	def param(self, name, value):
		self.declare_parameter(name, value)
		return self.get_parameter(name).get_parameter_value()

	def image_callback(self, msg):
		pass

def main(args=None):
	try:
		rclpy.init(args=args)
		node=TDMPCSocialNavigation()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		# 終了処理
		rclpy.shutdown()


if __name__ == '__main__':
	main()