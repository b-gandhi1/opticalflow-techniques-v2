#!/usr/bin/env python3.8

import cv2 as cv
import numpy as np
import time

DESIREDWIDTH = 640
DESIREDHEIGHT = 480

# read video
vid_path = 'data collection with franka/B07LabTrials/sample_10000_200.mp4'
cap = cv.VideoCapture(vid_path)

FPS = cap.get(cv.CAP_PROP_FPS)
tot_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# image enhancement parameters: 
contrast = 3
brightness = 5

for _ in range(tot_frames):
    ret, frame = cap.read()
    if not ret:
        print('ERROR: Cannot get frame.')
        break
    # frame = cv.resize(frame,(DESIREDWIDTH,DESIREDHEIGHT)) # resize frame to fit on screen
    
    # increase brightness 
    # bright_frame = np.int16(frame) + 100 
    # bright_frame = np.clip(bright_frame,0,255)
    # bright_frame = np.uint8(bright_frame)
    
    bright_frame = cv.addWeighted(frame, contrast, np.zeros(frame.shape, frame.dtype), 0, brightness)     
    
    cv.imshow('Brighter frame',bright_frame)
    
    if cv.waitKey(10) & 0xFF == ord('q'):
        print('Quitting...')
        break
    # time.sleep(1/FPS)
    
cap.release()
cv.destroyAllWindows()
