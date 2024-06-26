#!/usr/bin/env python3
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

from ultralytics import YOLO
import torch

import numpy as np

print("Imports done, loading model...")

model = YOLO('/home/prabhav/Desktop/IISc/yolov8m.pt')
 
print("Model loaded!")

def convert_nv12_to_rgb(frame_nv12, width, height):
  # Extract Y and UV planes from NV12 format
  Y = frame_nv12[:height, :width]
  UV = frame_nv12[height:, :width]

  # Reshape UV plane to the same size as Y
  U = UV[::2, ::2]
  V = UV[1::2, ::2]

  # Upsample U and V to the same size as Y
  U = cv2.resize(U, (width, height), interpolation=cv2.INTER_LINEAR)
  V = cv2.resize(V, (width, height), interpolation=cv2.INTER_LINEAR)

  # Create empty RGB frame
  frame_rgb = np.empty((height, width, 3), dtype=np.uint8)

  # YUV to RGB conversion
  frame_rgb[:,:,0] = np.clip(Y + 1.402 * (V - 128), 0, 255) # Red channel
  frame_rgb[:,:,1] = np.clip(Y - 0.344136 * (U - 128) - 0.714136 * (V - 128), 0, 255) # Green channel
  frame_rgb[:,:,2] = np.clip(Y + 1.772 * (U - 128), 0, 255) # Blue channel

  return frame_rgb

def nv12_to_rgb(y, uv, width, height):
  # Create empty RGB image
  rgb_img = np.zeros((height, width, 3), dtype=np.uint8)

  # Split the interleaved UV plane
  u = uv[:, 0::2].repeat(2, axis=0).repeat(2, axis=1)
  v = uv[:, 1::2].repeat(2, axis=0).repeat(2, axis=1)

  # YUV to RGB conversion
  c = y - 16
  d = u - 128
  e = v - 128

  r = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255)
  g = np.clip((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255)
  b = np.clip((298 * c + 516 * d + 128) >> 8, 0, 255)

  rgb_img[..., 0] = r
  rgb_img[..., 1] = g
  rgb_img[..., 2] = b

  return rgb_img

def callback(data):
 
  # Used to convert between ROS and OpenCV images
  #print("Bridge Created")
  br = CvBridge()
 
  # Output debugging information to the terminal
  rospy.loginfo("receiving video frame")
   
  # Convert ROS Image message to OpenCV image
  #print("Converting imgmsg to cv2")
  current_frame = br.imgmsg_to_cv2(data)

  print(type(current_frame))
  print(current_frame.shape, current_frame.dtype)

  # height, width = current_frame.shape[0], current_frame.shape[1]
  # rgb_frame = convert_nv12_to_rgb(current_frame, width, height)

  # width = 1024
  # height = 768

  # frame_size = width * height * 3 // 2
  # if current_frame.size:
  #   print(f"Error: Frame size {current_frame.size} does not match expected size {frame_size}")

  # y = np.frombuffer(current_frame[0:width*height], dtype=np.uint8).reshape((height, width))
  # uv = np.frombuffer(current_frame[width*height:], dtype=np.uint8).reshape((height // 2, width // 2 * 2))

  # Convert NV12 to RGB
  # rgb_frame = nv12_to_rgb(y, uv, width, height)

  #rgb_frame = np.zeros((768,1024,3), dtype="uint8")
  rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_YUV2BGR_NV12)

  #print("Running model on the image")
  #results = model.predict(current_frame, conf=0.5)

  #print(results)
   
  # Display image
  cv2.imshow("camera", rgb_frame)
   
  cv2.waitKey(1)
      
def receive_message():
 
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
  print("Initialization done")
  rospy.init_node('video_sub_py', anonymous=True)
   
  # Node is subscribing to the video_frames topic
  print("Subscriber Created")
  rospy.Subscriber('/hires_small_color', Image, callback)
 
  # spin() simply keeps python from exiting until this node is stopped
  print("Spinning")
  rospy.spin()
 
  # Close down the video stream when done
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  receive_message()
