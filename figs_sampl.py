import cv2 as cv

fib1_trans = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/fibrescope/transTz/fibrescope1-14-Feb-2024--19-22-46.mp4')
fib1_rot = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/fibrescope/rotation/fibrescope1-05-Feb-2024--16-54-31.mp4')
web1_rot = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/webcam/rotation/webcam1-05-Feb-2024--15-03-50.mp4')
web1_trans = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/webcam/transTz/webcam1-14-Feb-2024--20-01-16.mp4')

# get ref frame for all: 
ret1, fibR_frame_ref = fib1_rot.read()
ret2, fibT_frame_ref = fib1_trans.read()
ret3, webR_frame_ref = web1_rot.read()
ret4, webT_frame_ref = web1_trans.read()

# get mid frame: 

# get left frame: 

# get right frame:

# get up frame: 

# get down frame:

