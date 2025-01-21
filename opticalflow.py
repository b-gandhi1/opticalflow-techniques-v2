import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle # save 3D data from the different algorithms
import time
import os, re
import glob # this is used indirectly
import sys
import timeseries_test as tst
import matplotlib.pyplot as plt
import torch
import pandas as pd

# from pyOpticalFlow import getimgsfiles # from: pip install pyoptflow

# these are for video writer: 
FPS = 10.0 # 10 fps
# TOT_FRAMES = int(FPS*60) # 10 fps, 60 secs (1 min) long recording
# # TOT_FRAMES = int(FPS*5) # 5 secs
DESIREDWIDTH = 640
DESIREDHEIGHT = 480

# fibrescope image enhancement parameters: 
CONTRAST = 3
BRIGHTNESS = 5

inp_space = "" # global var

# def fibrescope_process(cap,frame):
def fibrescope_process(frame):

    # width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # frame = cv.resize(frame,(int(width/2),int(height/2)),)    
    kernel = np.ones((2,2),np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    x,y,w,h = 190,255,200,100 # after resizing frame size. 
    # rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    circle = cv.circle(mask_blank, (340,125), 100, (255,255,255), -1)
    masked = cv.bitwise_and(gray,gray,mask=circle)
    brightened = cv.addWeighted(masked, CONTRAST, np.zeros(masked.shape, masked.dtype), 0, BRIGHTNESS)     
    # binary = cv.threshold(brightened,57,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    # morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    # morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    # dilated = cv.dilate(morph_close,kernel)

    # masked = cv.flip(masked,1) # horizontal flip
    
    return masked

# def fish_undistort(ref_frame,checkboard):
#     # checkboard = imread()
#     # calibrate the camera

#     # return calib_params # calibration parameters, which will be used in fibrescope_process. 

#     pass 
# def webcam_process(cap,frame):
def webcam_process(frame):

    # width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # frame = cv.resize(frame,(int(width/2),int(height/2)),)

    kernel = np.ones((4,4),np.uint8)
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    x,y,w,h = 0,60,635,340 # (x,y) = top left params
    rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    masked = cv.bitwise_and(gray,gray,mask=rect)
    binary = cv.threshold(masked,50,255,cv.THRESH_BINARY)[1] 
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return masked 

def z_brightness(frame): # use this to get average brightness of each frame
    norm_frame = frame/np.max(frame)
    bright_avg = np.average(norm_frame)
    return bright_avg
    
def OF_LK(cap,ref_frame,img_process,savefilename,pitchroll,num): # Lucas-Kanade, sparse optical flow, local solution
    
    data_history = []
    
    # create video writer: 
    # root = "imu-fusion-outputs"
    # filename = "fib_vid_"+pitchroll+i+'.mp4'
    # vid_writer = cv.VideoWriter(os.path.join(root,filename),cv.VideoWriter_fourcc(*'mp4v'),FPS,(DESIREDWIDTH,DESIREDHEIGHT),0)
    
    # LK OF parameters: 
    if img_process == webcam_process:
        print('LK: Webcam')
        feature_params = dict( maxCorners = 700, 
                                qualityLevel = 0.15, # between 0 and 1. Lower numbers = higher quality level. 
                                minDistance = 25.0, # distance in pixels between points being monitored. 
                                blockSize = 5,
                                useHarrisDetector = False, 
                                k = 0.04 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense. 
        lk_params = dict( winSize  = (45, 45),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    elif img_process == fibrescope_process:
        print('LK: Fibrescope')
        
        if inp_space == "0-5":
            print('Spacing: 0-5')
            feature_params = dict( maxCorners = 300, 
                                    qualityLevel = 0.01, # between 0 and 1. Lower numbers = higher quality level. 
                                    minDistance = 5.0, # distance in pixels between points being monitored. 
                                    blockSize = 3,
                                    useHarrisDetector = False, # Shi-Tomasi better for corner detection than Harris for fibrescope. 
                                    k = 0.04 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense.
        elif inp_space == "1-0":
            print('Spacing: 1-0')
            feature_params = dict( maxCorners = 100, 
                                    qualityLevel = 0.01, # between 0 and 1. Lower numbers = higher quality level. 
                                    minDistance = 5.0, # distance in pixels between points being monitored. 
                                    blockSize = 3,
                                    useHarrisDetector = False, # Shi-Tomasi better for corner detection than Harris for fibrescope. 
                                    k = 0.04 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense.
        elif inp_space == "1-5":
            print('Spacing: 1-5')
            feature_params = dict( maxCorners = 50, 
                                qualityLevel = 0.01, # between 0 and 1. Lower numbers = higher quality level. 
                                minDistance = 5.0, # distance in pixels between points being monitored. 
                                blockSize = 3,
                                useHarrisDetector = False, # Shi-Tomasi better for corner detection than Harris for fibrescope. 
                                k = 0.04 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense.
        else: 
            print("ERROR: Please enter a valid argument for pin spacing.")
            exit()

        lk_params = dict( winSize = (45, 45),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    else:
        print("ERROR: Please enter a valid argument for imaging method used.")
        exit()
        
    # Parameters for lucas kanade optical flow
    # lk_params = dict( winSize  = (45, 45),
    #                 maxLevel = 2,
    #                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    color = np.random.randint(0, 255, (500, 3)) # Create some random colors 
    
    p0 = cv.goodFeaturesToTrack(ref_frame, mask = None, **feature_params) # Shi-Tomasi corner detection
    # cv.imshow('ref frame temp',ref_frame)
    mask_OF = np.zeros_like(ref_frame)

    p1,st,err = None,None,None
    x_val_store, y_val_store = [], []
    
    x_val_tensor, y_val_tensor = [],[] # torch.empty(), torch.empty() #torch.empty((0,2)), torch.empty((0,2)) # init tensors 
    z_val_store = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break 

        frame_filt = img_process(frame) # was: (cap,frame)
        # cv.imshow('FILTERED + CROPPED',frame_filt)
        
        # p1,st,err = cv.calcOpticalFlowPyrLK(ref_frame, frame_filt, p0, None, None, None,**lk_params)
        p1,st,err = cv.calcOpticalFlowPyrLK(ref_frame, frame_filt, p0, p1, st, err,**lk_params)
        z_val = z_brightness(frame_filt)
        magnitude, angle = cv.cartToPolar(p1[..., 0], p1[..., 1])
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask_OF = cv.line(mask_OF, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
            frame = cv.circle(frame_filt, (int(a), int(b)), 1, color[j].tolist(), -1)
        img = cv.add(frame_filt, mask_OF)
        cv.imshow('Optical Flow - Lucas-Kanade', img)
        
        # vid_writer.write(img) # write video
        
        # save data into a csv
        savedata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Get the current timestamp
        "magnitude": magnitude,
        "angle": angle,
        "x_val": p1[...,0],
        "x_val_1d": np.mean(p1[...,0]),
        "y_val": p1[...,1],
        "y_val_1d": np.mean(p1[...,1]),
        "z_val": z_val
        }
        data_history.append(savedata)
        
        # print("x val size: ", np.shape(p1[...,0]), " y val size: ", np.shape(p1[...,1]))
        
        # tests: 
        x_val, y_val = np.asarray(p1[...,0]), np.asarray(p1[...,1])
        # print('x_val size: ', np.shape(x_val), ' y_val size: ', np.shape(y_val))
        # print('x_mean: ', np.mean(x_val), ' y_mean: ', np.mean(y_val))
        x_val_store.append(np.mean(x_val))
        y_val_store.append(np.mean(y_val))
        
        # Tests conclusion = TIME TO MEAN!
        
        # tensors: 
        # x_val_tensor = torch.stack([x_val_tensor, torch.tensor(x_val)], dim=0)
        x_val_tensor.append(torch.tensor(x_val))
        # y_val_tensor = torch.stack([y_val_tensor, torch.tensor(y_val)], dim=0)
        y_val_tensor.append(torch.tensor(y_val))
        
        z_val_store.append(z_val) # np.append(z_val_store, z_val)
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            break
        
        # Use this for continuous differential measurement of OF: ---    
        # ref_frame = frame_filt.copy()
        # p0 = good_new.reshape(-1, 1, 2)
    
    # vid_writer.release()

    # test plots
    plt.figure()
    plt.plot(x_val_store, label='x_val')
    plt.legend()
    plt.figure()
    plt.plot(y_val_store, label='y_val')
    plt.legend()
    # tst.oneD_plots(x_val_store)
    # plt.title('x_val')
    # plt.figure()
    # tst.oneD_plots(y_val_store)
    # plt.title('y_val')
    plt.show()
    
    # with open(savefilename, 'wb+') as file: # filename needs to be 'sth.pkl'
    #     print("Saving data...")
    #     pickle.dump(data_history, file)
    print("Finished saving!")
    # save tensors: 
    x_val_tensor = torch.stack(x_val_tensor, dim=0)
    y_val_tensor = torch.stack(y_val_tensor, dim=0)
    ptfiles = os.path.dirname(savefilename) # directory of savefilename
    # num = max((int(num) for file in ptfiles for num in re.findall(r'\d+', os.path.splitext(file)[0])), default=0)+1
    torch.save(x_val_tensor, ptfiles+'/tensor_x_val'+pitchroll+str(num)+'.pt')
    torch.save(y_val_tensor, ptfiles+'/tensor_y_val'+pitchroll+str(num)+'.pt')
    LK_Zavg = pd.DataFrame({"x_vals":x_val_store, "y_vals":y_val_store, "z_vals":z_val_store}, dtype=float)
    LK_Zavg.to_csv(ptfiles+'/skins-'+pitchroll+str(num)+'.csv') # save csv file for z vals

# def OF_GF(cap,ref_frame,img_process,savefilename,mask): # Gunnar-Farneback, dense optical flow
    # keeps getting killed... not sure why. 
    # mask[...,1] = 255 # saturation

    # data_history = []
    
    # while True:
    #     ret, frame = cap.read()
    #     if not ret: break
    
    #     frame_filt = img_process(frame) # was: (cap,frame)
    #     flow = cv.calcOpticalFlowFarneback(ref_frame,frame_filt,None,pyr_scale=0.5,levels=2,winsize=3,iterations=2,poly_n=5,poly_sigma=1.1,flags=0) # what do these parameters mean?? 
    #     # explain - https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html 
    #     mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #     # mask[..., 0] = ang*180/np.pi/2
    #     mask[...,0] = np.rad2deg(ang) # ERROR HERE
    #     mask[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    #     # Converts HSV to RGB (BGR) color representation
    #     rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    #     # Opens a new window and displays the output frame
    #     cv.imshow('Optical Flow - Gunnar Farneback', rgb)

    #     # save data into a csv
    #     savedata = {
    #     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Get the current timestamp
    #     "data": [mag,ang] }
    #     data_history.append(savedata)

    #     if cv.waitKey(10) & 0xFF == ord('q'):
    #         print('Quitting...')
    #         break
    #     # ref_frame = frame_filt # use this for continuous differential measurement of OF

    # with open(savefilename, 'wb+') as file: # filename needs to be 'sth.pkl'
    #     pickle.dump(data_history, file)
    
def blobdetect(cap,img_process,savefilename):

    params = cv.SimpleBlobDetector_Params() # create blob detector
    params.filterByColor = True # changs min and max thresholds for tuning. 
    if img_process == fibrescope_process:
        params.minThreshold = 50
    if img_process == webcam_process:
        params.minThreshold = 30
    # params.minThreshold = ... # 50 # 120 
    params.maxThreshold = 120 
    params.blobColor = 255
    params.filterByArea = False
    params.filterByCircularity = False
    params.minCircularity = 0.5 # works for grayscale webcam
    params.filterByConvexity = False
    params.minConvexity = 0.87
    params.filterByInertia = False
    params.minInertiaRatio = 0.7
    detector = cv.SimpleBlobDetector_create(params) # create blob detector

    data_history = []
    x_val_store, y_val_store = [], []
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = img_process(frame) # binarize frame. was: (cap,frame)
    
        keypoints = detector.detect(frame) # detect blobs
        centroids = np.array([keypoint.pt for keypoint in keypoints]) # extract centroids of blobs
        
        # if centroids is empty, diff_x and diff_y are zeros
        if centroids.size == 0:
            diff_x, diff_y = 0, 0
        else:
            diff_x = np.diff(centroids[:,0])
            diff_y = np.diff(centroids[:,1])
        # simple euclidean dist, max
        # diff_x = np.diff(centroids[:,0])
        # diff_y = np.diff(centroids[:,1])
        # dist_mag = np.sum([diff_x**2+diff_y**2],axis=0)
        magnitude, angle = cv.cartToPolar(diff_x, diff_y)
        z_val = z_brightness(frame)
        
        BD_img = cv.drawKeypoints(frame,keypoints,np.array([]),(0,0,255),cv.DRAW_MATCHES_FLAGS_DEFAULT)
        cv.imshow('OpenCV: Blob Detection',BD_img)

        # save data in a pickle file
        savedata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Get the current timestamp
        "magnitude": magnitude, 
        "angle": angle,
        "x_val": diff_x, 
        "x_val_1d": np.median(diff_x),
        "y_val": diff_y,
        "y_val_1d": np.median(diff_y),
        "z_val": z_val 
        }
        data_history.append(savedata)
        
        # tests: 
        x_val, y_val = np.median(diff_x), np.median(diff_y)
        # print('x_val: ', x_val, 'y_val: ', y_val, 'z_val: ', z_val)
        x_val_store.append(x_val)
        y_val_store.append(y_val)
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            break
    
    # Tests plots: 
    # tst.oneD_plots(x_val_store)
    # plt.title('x_val')
    # tst.oneD_plots(y_val_store)
    # plt.title('y_val')
    # plt.show()
    
    with open(savefilename, 'wb+') as file: # filename needs to be 'sth.pkl'
        pickle.dump(data_history, file)

def main(img_process_selector,loadpath,inp_path,pitchroll,i):
    # read video opencv
    # video_path = input('Enter input video path: ')
    cap = cv.VideoCapture(loadpath) # insert video path
    if not cap.isOpened(): print("ERROR: Cannot open camera/video file.")

    # # set size
    # cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
    
    # take reference frame 
    if img_process_selector == 'w':
        cap.set(cv.CAP_PROP_POS_FRAMES, 13) # since first ref frame is messy for some reason.. does not cover all pins in binary verison. 

    ret, ref_frame = cap.read()
    if not ret: 
        print('STATUS: End of frames OR Cannot get frame.') 
        pass
    
    # if img_process_selector == 'f':
    #     ref_frame = cv.flip(ref_frame,1) # flip horizontal

    cap.set(cv.CAP_PROP_POS_FRAMES, 0) # reset back. 
    
    # width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # ref_frame = cv.resize(ref_frame,(int(width/2),int(height/2)),)
    mask_GF = np.zeros_like(ref_frame) # parameter for GF

    # user input: select webcam or fibrescope
    switcher = {
        'f': fibrescope_process,
        'w': webcam_process
    }
    # img_process_selector = input("Select process, 'w' for webcam, and 'f' for fibrescope: ")
    print('Process selected: ', img_process_selector)
    img_process = switcher.get(img_process_selector)
    cv.imshow('reference frame before filtering',ref_frame)
    ref_frame = img_process(ref_frame) # was: (cap,ref_frame)
    cv.imshow('reference frame after filtering',ref_frame)
    # filenames to save output data: 
    savefilename_LK = os.path.join('skins-test-outputs', inp_path+'_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.pkl')
    # savefilename_LK = 'OF_outputs/trial.pkl'
    # savefilename_GF = os.path.join('OF_outputs','GF_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.pkl')
    # savefilename_BD = os.path.join('OF_outputs/data4_feb2024','BD_binary_web_'+time.strftime("%Y-%m-%d_%H-%M-%S")+'.pkl')

    # # multiprocess the 4 methods together
    # OF_LK_process = mp.Process(target=OF_LK, args=(cap,ref_frame,img_process,savefilename_LK))
    # OF_GF_process = mp.Process(target=OF_GF, args=(cap,ref_frame,img_process,savefilename_GF,mask_GF))
    # # OF_HS_process = mp.Process(target=OF_HS, args=(cap,ref_frame,img_process))
    # blobdetect_process = mp.Process(target=blobdetect, args=(cap,img_process,savefilename_BD))

    try: 
        # # start multi-processes
        # OF_LK_process.start()
        # OF_GF_process.start()
        # # OF_HS_process.start()
        # blobdetect_process.start()

        # # finish multi together
        # OF_LK_process.join()
        # OF_GF_process.join()
        # # OF_HS_process.join()
        # blobdetect_process.join()
        # print('All processes have finished.')
        
        # not using multi-processing: 
        OF_LK(cap,ref_frame,img_process,savefilename_LK,pitchroll,i)
        
        # take reference frame - reset cap frame number. 
        if img_process_selector == 'w':
            cap.set(cv.CAP_PROP_POS_FRAMES, 3) # since first ref frame is messy for some reason.. does not cover all pins in binary verison. 
        ret, ref_frame = cap.read()
        if not ret: 
            print('STATUS: End of frames OR Cannot get frame.') 
            pass
        cap.set(cv.CAP_PROP_POS_FRAMES, 0) # reset back. 
        
        # OF_GF(cap,ref_frame,img_process,savefilename_GF,mask_GF) # keeps gettig killed, due to RAM and CPU being saturated.
        # blobdetect(cap,img_process,savefilename_BD)

    except KeyboardInterrupt:
        print('*****ERROR: Manually interrupted*****')
        pass

    cap.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
    
    img_process_selector = "f" #sys.argv[1] # f or w
    
    inp_path = sys.argv[1] # pitchN or rollN where N = 1:8
    
    inp_space = sys.argv[2] # spacing: 0-5 | 1-0 | 1-5
    
    # eg command: python opticalflow.py f pitch1
    
    length = len(inp_path)
    pitchroll, i = inp_path[:length-1], inp_path[length-1] # split number at the endfrom filename 

    pitchroll_path = glob.glob("data_collection_with_franka/B07LabTrials/skins-data/cm"+inp_space+"/"+pitchroll+"/fibrescope-*"+i+".mp4")[0]

    # if pitchroll == "pitch":
    #     pitchroll_path = "skins-data/cm"+inp_space+pitchroll+"/fibrescope"+i
    # elif pitchroll == "roll":
    #     pitchroll_path = "roll_22-aug-2024/fibrescope"+i
    # else:
    #     print("ERROR: Unrecognised input for pressure selector.")
    
    inp_path_full = pitchroll_path
    
    main(img_process_selector,inp_path_full,inp_path,pitchroll,i)
    