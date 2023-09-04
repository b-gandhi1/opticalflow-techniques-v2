#!/usr/bin/python
# -*- coding: utf-8 -*- 

import cv2 as cv
import numpy as np
import os
import glob

def fibrescope_process(frame):
    # undistorting image: 
    balance = 0.7 # change this, high val will give wider view: 0 < balance <= 1 
    dim2 = None
    dim3 = None

    DIM,K,D = fish_calib_params()
    
    # for p in sys.argv[1:]:
    img = frame
    dim1 = img.shape[:2][::-1] # dim1 = dim of raw img to undistort
    
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

    if not dim2:
        dim2 = dim1    
    if not dim3:
        dim3 = dim1    
        
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)

    map1,map2 = cv.fisheye.initUndistortRectifyMap(scaled_K,D,np.eye(3),new_K,dim3,cv.CV_16SC2)
    undistorted_img = cv.remap(img,map1,map2,interpolation=cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT)

    cv.imshow('undistorted img', undistorted_img)
    cv.waitKey(0)

    frame = undistorted_img # apply undistortion to frame

    # continue regular img processing: binarize, morph 
    kernel = np.ones((3,3),np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_RBG2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    # x,y,w,h = 140,70,200,150 # after resizing frame size. UPDATE THIS
    x,y,w,h = 350,280,200,110 # after resizing frame size. UPDATE THIS
    rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    masked = cv.bitwise_and(gray,gray,mask=rect)
    binary = cv.threshold(masked,40,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return dilated

def fish_calib_params():

    CHECKERBOARD = (7,7) # number of inside corners in their chessboard, UPDATE!

    # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0 # part 1
    # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f # part 2
    
    subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.images = glob.glob('*.jpg')for fname in images:
    
    # load calibration images - checkerboard
    images = glob.glob('data collection with franka/ViconLab/fibrescope/calibrationcheckboard/calib*.jpg') # define path

    for fname in images:
        img = cv.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."    
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    DIM = _img_shape[::-1]
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    
    return DIM,K,D
    
def webcam_process(frame):
    kernel = np.ones((5,5),np.uint8)
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    x,y,w,h = 75,50,845,430 # (x,y) = top left params
    rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    masked = cv.bitwise_and(gray,gray,mask=rect)
    binary = cv.threshold(masked,50,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1] # might remove: + cv.thresh_otsu
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return dilated 

def main():
    # read video opencv
    video_path = input('Enter input video path: ')
    cap = cv.VideoCapture(video_path) # insert video path
    if not cap.isOpened(): print("ERROR: Cannot open camera/video file.")

    # # set size
    # cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)

    # get size
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # take reference frame 
    ret, ref_frame = cap.read()

    # resize:
    ref_frame = cv.resize(ref_frame,(int(width/2),int(height/2)),)
    
    # user input: select webcam or fibrescope
    switcher = {
        'f': fibrescope_process,
        'w': webcam_process
    }
    img_process_selector = input("Select process, 'w' for webcam, and 'f' for fibrescope: ")
    print('Process selected: ', img_process_selector)
    img_process = switcher.get(img_process_selector)
    ref_frame_filt = img_process(ref_frame)
    
    # cv.imshow('Raw frame',ref_frame)
    # cv.imshow('Filtered frame', ref_frame_filt)
    
    while True: 
        ret, frame = cap.read()
        if not ret: break
        # frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) # convert ref_frame to grayscale
        frame = cv.resize(frame,(int(width/2),int(height/2)),)
        cv.imshow('Raw',frame)
        frame_filt = img_process(frame)
        cv.imshow('Filtered img from '+img_process_selector,frame_filt)
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            break

    # if cv.waitKey(0) & 0xFF == ord('q'):
    #     print('Quitting...')
    #     cap.release()
    #     cv.destroyAllWindows()

if __name__ == '__main__':
    main()
    