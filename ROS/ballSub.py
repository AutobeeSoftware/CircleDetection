#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:34:53 2022

@author: autobee2021
"""

import rospy
from borc_package.msg import BallLocation
import RPi.GPIO as GPIO
import message_filters
from std_msgs.msg import Bool
#600 falan sol
GPIO.setmode(GPIO.BOARD)
GPIO.setup(32,GPIO.OUT) #left
GPIO.setup(33,GPIO.OUT) #right
left_pwm = GPIO.PWM(32,50)
right_pwm = GPIO.PWM(33,50)
left_pwm.start(7.5)
right_pwm.start(7.5)

def follow_the_ball(mesaj, ball):
    rospy.loginfo("The location of ball: %s"%mesaj.location)
    rospy.loginfo("As")
    rospy.loginfo(ball)
    if ball:
        if 300 < mesaj.location < 340:
            right_pwm.ChangeDutyCycle(9)
            left_pwm.ChangeDutyCycle(9)
            rospy.loginfo("Duz gidiyor")
        elif mesaj.location >= 340:
            right_pwm.ChangeDutyCycle(9)
            left_pwm.ChangeDutyCycle(6)
            rospy.loginfo("saga donuyor")
        elif mesaj.location <= 300:
            right_pwm.ChangeDutyCycle(6)
            left_pwm.ChangeDutyCycle(9)
            rospy.loginfo("Sola donuyor")
    else:
        left_pwm.ChangeDutyCycle(8)
        right_pwm.ChangeDutyCycle(6)
        
def ball_subscriber(*args, **kwargs):
    rospy.init_node("point_of_ball")
    location_sub = message_filters.Subscriber("location_of_ball", BallLocation, follow_the_ball)
    isthere_sub = message_filters.Subscriber("ball_is_there", Bool, follow_the_ball)
    rospy.spin()
    
ball_subscriber()
