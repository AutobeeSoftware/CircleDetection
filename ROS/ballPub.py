#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:59:36 2022

@author: autobee2021
"""
import rospy
import cv2
from cv_bridge import CvBridge
#import imutils
import numpy as np
from borc_package.msg import BallLocation
from std_msgs.msg import Bool


camera_id = "/dev/video0"
cam = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
print(cam.isOpened())
bridge=CvBridge()


def ball_publisher():
    rospy.init_node("publish_the_ball")
    pub1 = rospy.Publisher("location_of_ball", BallLocation, queue_size=1)
    pub2 = rospy.Publisher("ball_is_there", Bool, queue_size=1)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        
        _, im = cam.read()
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        green_light=np.array([0, 155, 149],np.uint8)
        green_dark=np.array([180,255,255],np.uint8)
        green=cv2.inRange(hsv,green_light,green_dark) 
        kernel=np.ones((15,15),"uint8")
        green=cv2.dilate(green,kernel)
        res_green = cv2.bitwise_and(im,im, mask=green)
        
        (contours,hierarchy)=cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for _,c in enumerate(contours):
            area=cv2.contourArea(c)
            if area>3000:
                found = True
                cv2.drawContours(im,c,-1,(0,255,0),2)
                M = cv2.moments(c)

                cx = int(M["m10"]/ M["m00"])
                cy = int(M["m01"]/ M["m00"])

                cv2.circle(im,(cx,cy),7,(255,255,255),-1)
                rospy.loginfo(cx)
                rospy.loginfo(found)
                pub1.publish(cx)
                pub2.publish(found)
                
        
        
if __name__ == '__main__':
    try:
        if ball_publisher():
            ball_publisher()
        else:
            cam.release()
		
    except rospy.ROSInterruptException:
        pass


                
        
