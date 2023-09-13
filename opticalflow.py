#!/usr/bin/python
# -*- coding: utf-8 -*- 

import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle # save 3D data from the different algorithms
import time
import os
import glob

# from pyOpticalFlow import getimgsfiles # from: pip install pyoptflow

def fibrescope_process(cap,frame):

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    frame = cv.resize(frame,(int(width/2),int(height/2)),)

    kernel = np.ones((3,3),np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    x,y,w,h = 350,280,200,110 # after resizing frame size. 
    rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    masked = cv.bitwise_and(gray,gray,mask=rect)
    binary = cv.threshold(masked,10,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return dilated

def fish_undistort(ref_frame,checkboard):
    # checkboard = imread()
    # calibrate the camera

    # return calib_params # calibration parameters, which will be used in fibrescope_process. 

    pass 
def webcam_process(cap,frame):

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    frame = cv.resize(frame,(int(width/2),int(height/2)),)

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
def OF_LK(cap,ref_frame,img_process,savefilename): # Lucas-Kanade, sparse optical flow, local solution
    
    data_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break 

        frame_filt = img_process(cap,frame)
        # cv.imshow('FILTERED + CROPPED',frame_filt)

        # LK OF parameters: 
        feature_params = dict( maxCorners = 100, # 100 max val, and works best
                                    qualityLevel = 0.01, # between 0 and 1. Lower numbers = higher quality level. 
                                    minDistance = 5, # distance in pixels between points being monitored. 
                                    blockSize = 5 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense. 

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (45, 45),
                        maxLevel = 2,
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) 

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))


        p0 = cv.goodFeaturesToTrack(ref_frame, mask = None, **feature_params)
        mask_OF = np.zeros_like(ref_frame)
        
        p1,st,err = cv.calcOpticalFlowPyrLK(ref_frame, frame_filt, p0, None, None, None,**lk_params)
        magnitude, angle = cv.cartToPolar(p1[..., 0], p1[..., 1])
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask_OF = cv.line(mask_OF, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame_filt, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame_filt, mask_OF)
        cv.imshow('Optical Flow - Lucas-Kanade', img)

        # save data into a csv
        savedata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Get the current timestamp
        "data": [magnitude,angle] }
        data_history.append(savedata)

        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            break
        
        # Use this for continuous differential measurement of OF: ---    
        # ref_frame = frame_filt.copy()
        # p0 = good_new.reshape(-1, 1, 2)

    with open(savefilename, 'wb+') as file: # filename needs to be 'sth.pkl'
        pickle.dump(data_history, file)

def OF_GF(cap,ref_frame,img_process,savefilename,mask): # Gunnar-Farneback, dense optical flow
    # keeps getting killed... not sure why. 
    mask[...,1] = 255 # saturation

    data_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_filt = img_process(cap,frame)
        flow = cv.calcOpticalFlowFarneback(ref_frame,frame_filt,None,pyr_scale=0.5,levels=2,winsize=3,iterations=2,poly_n=5,poly_sigma=1.1,flags=0) # what do these parameters mean?? 
        # explain - https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html 
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # mask[..., 0] = ang*180/np.pi/2
        mask[...,0] = np.rad2deg(ang) # ERROR HERE
        mask[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Opens a new window and displays the output frame
        cv.imshow('Optical Flow - Gunnar Farneback', rgb)

        # save data into a csv
        savedata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Get the current timestamp
        "data": [mag,ang] }
        data_history.append(savedata)

        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            break
        # ref_frame = frame_filt # use this for continuous differential measurement of OF

    with open(savefilename, 'wb+') as file: # filename needs to be 'sth.pkl'
        pickle.dump(data_history, file)
    
def OF_HS(cap,ref_frame,img_process,savefilename): # Horn-Schunk, global solution, smoother, brightness constancy. 
    pass
def blobdetect(cap,img_process,savefilename):

    params = cv.SimpleBlobDetector_Params() # create blob detector
    params.filterByColor = True # changs min and max thresholds for tuning. 
    params.minThreshold = 90 
    params.maxThreshold = 170 
    params.blobColor = 255
    params.filterByArea = False
    params.filterByCircularity = False
    params.minCircularity = 0.8
    params.filterByConvexity = False
    params.minConvexity = 0.87
    params.filterByInertia = False
    params.minInertiaRatio = 0.7
    detector = cv.SimpleBlobDetector_create(params) # create blob detector

    data_history = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = img_process(cap,frame) # binarize frame
    
        keypoints = detector.detect(frame) # detect blobs
        centroids = np.array([keypoint.pt for keypoint in keypoints]) # extract centroids of blobs
    
        # simple euclidean dist, max
        diff_x = np.diff(centroids[:,0])
        diff_y = np.diff(centroids[:,1])
        dist_mag = np.sum([diff_x**2+diff_y**2],axis=0)

        BD_img = cv.drawKeypoints(frame,keypoints,np.array([]),(0,0,255),cv.DRAW_MATCHES_FLAGS_DEFAULT)
        cv.imshow('OpenCV: Blob Detection',BD_img)

        # save data in a pickle file
        savedata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Get the current timestamp
        "data": dist_mag }
        data_history.append(savedata)            

        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            break

    with open(savefilename, 'wb+') as file: # filename needs to be 'sth.pkl'
        pickle.dump(data_history, file)



def main():
    # read video opencv
    video_path = input('Enter input video path: ')
    cap = cv.VideoCapture(video_path) # insert video path
    if not cap.isOpened(): print("ERROR: Cannot open camera/video file.")

    # # set size
    # cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
    
    # take reference frame 
    ret, ref_frame = cap.read()
    if not ret: print('ERROR: Cannot get frame.')
    
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    ref_frame = cv.resize(ref_frame,(int(width/2),int(height/2)),)
    mask_GF = np.zeros_like(ref_frame) # parameter for GF

    # user input: select webcam or fibrescope
    switcher = {
        'f': fibrescope_process,
        'w': webcam_process
    }
    img_process_selector = input("Select process, 'w' for webcam, and 'f' for fibrescope: ")
    print('Process selected: ', img_process_selector)
    img_process = switcher.get(img_process_selector)
    cv.imshow('reference frame before filtering',ref_frame)
    ref_frame = img_process(cap,ref_frame)
    cv.imshow('reference frame after filtering',ref_frame)
    # filenames to save output data: 
    savefilename_LK = input('Enter filename to save the current trial for OF_LK:')
    savefilename_GF = input('Enter filename to save the current trial for OF_GF:')
    savefilename_BD = input('Enter filename to save the current trial for blobdetect:')
    # ensure path = outputs/filenames.pkl
    savefilename_LK = 'outputs/'+savefilename_LK+'.pkl'
    savefilename_GF = 'outputs/'+savefilename_GF+'.pkl'
    savefilename_BD = 'outputs/'+savefilename_BD+'.pkl'

    # multiprocess the 4 methods together
    OF_LK_process = mp.Process(target=OF_LK, args=(cap,ref_frame,img_process,savefilename_LK))
    OF_GF_process = mp.Process(target=OF_GF, args=(cap,ref_frame,img_process,savefilename_GF,mask_GF))
    # OF_HS_process = mp.Process(target=OF_HS, args=(cap,ref_frame,img_process))
    blobdetect_process = mp.Process(target=blobdetect, args=(cap,img_process,savefilename_BD))

    try: 
        # start multi-processes
        OF_LK_process.start()
        OF_GF_process.start()
        # OF_HS_process.start()
        blobdetect_process.start()

        # finish multi together
        OF_LK_process.join()
        OF_GF_process.join()
        # OF_HS_process.join()
        blobdetect_process.join()
        print('All processes have finished.')
        
        # not using multi-processing: 
        # OF_LK(cap,ref_frame,img_process,savefilename_LK)
        # OF_GF(cap,ref_frame,img_process,savefilename_GF,mask_GF) # keeps gettig killed, due to RAM and CPU being saturated.
        # blobdetect(cap,img_process,savefilename_BD)

    except KeyboardInterrupt:
        print('*****ERROR: Manually interrupted*****')
        pass

    cap.release()
    cv.destroyAllWindows()
if __name__ == '__main__':
    main()