import cv2 as cv
import numpy as np
import os
import glob
import datetime
import time
import sys

WIDTH = 640
HEIGHT = 480

# fibrescope image enhancement parameters: 
CONTRAST = 3
BRIGHTNESS = 5
# def fibrescope_process(frame,width,height,map1,map2):    
def fibrescope_process(frame):    
    # undistorted_img = cv.remap(frame,map1,map2,interpolation=cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT) # fisheye undistort

    # cv.imshow('calib: raw img, press key to close', img)
    # print('raw img', img.shape)
    # cv.imshow('calib: undistorted img, press key to close', undistorted_img.shape)
    # print('undistorted img', undistorted_img)
    # cv.waitKey(0)

    # frame = undistorted_img # apply undistortion to frame
    
    # continue regular img processing: binarize, morph 
    # frame = cv.resize(frame,(int(width/2),int(height/2))) # resize img after undistortion
    kernel = np.ones((2,2),np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    # x,y,w,h = 140,70,200,150 # after resizing frame size. UPDATE THIS
    # x,y,w,h = 350,280,200,110 # after resizing frame size. UPDATE THIS
    # x,y,w,h = 0,0,frame.shape[1],frame.shape[0] # no cropping, using all of it for now. 
    # rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    circle = cv.circle(mask_blank, (340,125), 100, (255,255,255), -1)
    masked = cv.bitwise_and(gray,gray,mask=circle)
    brightened = cv.addWeighted(masked, CONTRAST, np.zeros(masked.shape, masked.dtype), 0, BRIGHTNESS)     
    binary = cv.threshold(brightened,55,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    # eroded = cv.erode(binary,np.ones((4,3),np.uint8))
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    masked = cv.flip(masked,1) # horizontal flip 

    return masked

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
    images = glob.glob('data collection with franka/ViconLab/fibrescope/calibrationcheckboard/calib*.png') # define path

    for fname in images:
        img = cv.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."    
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        
        # temp show images: 
        cv.imshow('calibration images', gray)
        print(np.shape(gray))
        # use cv.waitkey to set a timer for 3 sec
        cv.waitKey(300)
        cv.destroyWindow('calibration images')
        
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
    
    # save return parameters: 
    np.savez('data collection with franka/ViconLab/fibrescope/CalibrationParameters_DIM_K_D_'+str(datetime.date.today())+'.npz',DIM=DIM,K=K,D=D)
    # another function is np.savez_compressed() - what is the difference?? 
    # return DIM,K,D
    
# def webcam_process(frame,width,height,a,b): # a,b blank parameters
def webcam_process(frame):
    # frame = cv.resize(frame,(int(width/2),int(height/2)))
    kernel = np.ones((4,4),np.uint8)
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    x,y,w,h = 0,60,635,340 # (x,y) = top left params
    rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    masked = cv.bitwise_and(gray,gray,mask=rect)
    binary = cv.threshold(masked,231,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    # same threshold does not work for static image and video both. Static img thresh = 50, video thresh = 140. 
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return dilated 

def main(img_process_selector,path):
    # read video opencv
    # video_path = input('Enter input video path: ')
    cap = cv.VideoCapture(path) # insert video path
    if not cap.isOpened(): print("ERROR: Cannot open camera/video file.") 
    # # set size
    # cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)

    # get size
    # width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv.CAP_PROP_FPS) # get FPS
    tot_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) # get tot frames
    time_len = tot_frames/fps
    cap.set(cv.CAP_PROP_FPS, fps)
    print('FPS: ',fps)
    print('Total frames: ',tot_frames)
    print('Time length: ',time_len)
    
    # take reference frame 
    ret, ref_frame = cap.read()

    # get size:
    width = np.shape(ref_frame)[1]
    height = np.shape(ref_frame)[0]
    
    print('Size (width,height): ',width, ' ', height)
    # resize:
    # ref_frame = cv.resize(ref_frame,(int(width/2),int(height/2)))
    
    # fibrescope calibration parameters -----
    
    # recalib = input('Would you like to re-calibrate today? (Y/N | y/n): ')
    # if recalib == 'y' or recalib == 'Y': 
    #     print('Re-calibrating...')
    #     # run calibration with checkerboard calib images
    #     DIM,K,D = fish_calib_params()
    # elif recalib == 'n' or recalib == 'N':
    #     print('Using saved calibration parameters.')
    #     # read calib params from file
    #     # calib_path = input('Enter calibration parameters file path: ')
    #     calib_path = 'data collection with franka/ViconLab/fibrescope/CalibrationParameters_DIM_K_D_2023-09-12.npz'
    #     npzfile = np.load(calib_path)
    
    #     # verify variable names
    #     variable_names = npzfile.files

    #     # Print the variable names
    #     for variable_name in variable_names:
    #         print(variable_name)
    
    #     DIM = npzfile['DIM']
    #     K = npzfile['K']
    #     D = npzfile['D']
    # else:
    #     print('ERROR: Invalid input.')
    #     exit()
    
    # map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv.CV_16SC2)

    # --------
    
    # user input: select webcam or fibrescope
    switcher = {
        'f': fibrescope_process,
        'w': webcam_process
    }
    # img_process_selector = input("Select process, 'w' for webcam, and 'f' for fibrescope: ")
    print('Process selected: ', img_process_selector)
    img_process = switcher.get(img_process_selector)
    
    # ref_frame_filt = img_process(ref_frame,width,height,map1,map2)
    ref_frame_filt = img_process(ref_frame)
    
    cv.imshow('Raw ref_frame, press key to exit',cv.cvtColor(ref_frame,cv.COLOR_RGB2GRAY))
    cv.imshow('Filtered ref_frame, press key to exit', ref_frame_filt)
    cv.waitKey(0)
    
    # start = time.time()
    
    for count in range(tot_frames): 
        ret, frame = cap.read()
        if not ret: break
        cv.imshow('Raw video',frame)
        # frame_filt = img_process(frame,width,height,map1,map2)
        frame_filt = img_process(frame)
        cv.imshow('Filtered video from '+img_process_selector,frame_filt)
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            # cap.release()
            # cv.destroyAllWindows()
            break
        # elapsed_time = time.gmtime(time.time()-start)
        # elapsed_time_format = "{:02d}:{:02d}".format(elapsed_time.tm_min,elapsed_time.tm_sec)
        # print('Elapsed time: ',elapsed_time_format)
        # time.sleep(1/fps)
        # imshow the video is a lot quicker than the actual video but i am leaving it as is as this is not important right now. 

    cap.release()
    cv.destroyAllWindows()
    print('Total frames = ', count)
    # print('Tot frames should be = ', fps*60)
if __name__ == '__main__':
    img_process_selector = sys.argv[1]
    path = sys.argv[2]
    main(img_process_selector,path)