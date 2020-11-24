import numpy as np
import cv2
from tensorflow import keras


frameWidth = 1080 #640  # CAMERA RESOLUTION
frameHeight = 1920#480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture("http://192.168.8.176:4747/video")
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
cap.set(5, 1)
while True:
    success, img = cap.read() # Returns a bool (True/False). If frame is read correctly,
    cv2.imshow("Processed Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Code to stop camera workin
     break