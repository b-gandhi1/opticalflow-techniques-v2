import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt # use this for quiver plots

# fibrescope image enhancement parameters: 
CONTRAST = 3
BRIGHTNESS = 5
def fibrescope_process(frame, centre):

    # width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # frame = cv.resize(frame,(int(width/2),int(height/2)),)

    kernel = np.ones((2,2),np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    # x,y,w,h = 350,280,200,110 # after resizing frame size. 
    # rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    circle = cv.circle(mask_blank, centre, 100, (255,255,255), -1)
    masked = cv.bitwise_and(gray,gray,mask=circle)
    brightened = cv.addWeighted(masked, CONTRAST, np.zeros(masked.shape, masked.dtype), 0, BRIGHTNESS)     
    # binary = cv.threshold(brightened,57,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    # morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    # morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    # dilated = cv.dilate(morph_close,kernel)

    return brightened

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
    # binary = cv.threshold(masked,50,255,cv.THRESH_BINARY)[1] 
    # morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    # morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    # dilated = cv.dilate(morph_close,kernel)

    return masked

def OF_LK(frame,ref_frame,img_process): # Lucas-Kanade, sparse optical flow, local solution
        
    # LK OF parameters: 
    if img_process == 'w':
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
    
    elif img_process == 'f':
        print('LK: Fibrescope')
        feature_params = dict( maxCorners = 100, 
                                qualityLevel = 0.01, # between 0 and 1. Lower numbers = higher quality level. 
                                minDistance = 5.0, # distance in pixels between points being monitored. 
                                blockSize = 3,
                                useHarrisDetector = False, # Shi-Tomasi better for corner detection than Harris for fibrescope. 
                                k = 0.04 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense.
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
    # p0 = cv.cornerHarris(ref_frame, 10,10,0.3) # Harris corner detection, ERROR. figure out how to use this!! 
    # cv.imshow('ref frame temp',ref_frame)
    mask_OF = np.zeros_like(ref_frame)

    p1,st,err = None,None,None
    
    # while True:

    frame_filt = img_process(frame) # was: (cap,frame)
    # cv.imshow('FILTERED + CROPPED',frame_filt)
    
    # p1,st,err = cv.calcOpticalFlowPyrLK(ref_frame, frame_filt, p0, None, None, None,**lk_params)
    p1,st,err = cv.calcOpticalFlowPyrLK(ref_frame, frame_filt, p0, p1, st, err,**lk_params)
    magnitude, angle = cv.cartToPolar(p1[..., 0], p1[..., 1])
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask_OF = cv.line(mask_OF, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame_filt = cv.circle(frame_filt, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame_filt, mask_OF)
    # cv.imshow('Optical Flow - Lucas-Kanade', img)        
    return img, p1
    
def main():
    fib1_trans = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/fibrescope/transTz/fibrescope1-14-Feb-2024--19-22-46.mp4')
    fib1_rot = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/fibrescope/rotation/fibrescope1-05-Feb-2024--16-54-31.mp4')
    web1_rot = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/webcam/rotation/webcam1-05-Feb-2024--15-03-50.mp4')
    web1_trans = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/webcam/transTz/webcam1-14-Feb-2024--20-01-16.mp4')

    # get mid/down frame for all: 
    ret1, fib_mid = fib1_rot.read()
    ret2, fib_down = fib1_trans.read()
    web1_rot.set(cv.CAP_PROP_POS_FRAMES, 2) 
    ret3, web_mid = web1_rot.read()
    web1_trans.set(cv.CAP_PROP_POS_FRAMES, 2)
    ret4, web_down = web1_trans.read()

    if not ret1 or not ret2 or not ret3 or not ret4: print('ERROR: Cannot get frame.')
    
    fib_rot_centre = (355,325)
    fib_trans_centre = (420,300)
    
    # get mid/down frame: 
    fib_mid = fibrescope_process(fib_mid, fib_rot_centre)
    fib_down = fibrescope_process(fib_down, fib_trans_centre)
    web_mid = webcam_process(web_mid)
    web_down = webcam_process(web_down)
    
    # show frames and save
    cv.imshow('fib rot ref',fib_mid)
    cv.imshow('fib trans ref',fib_down)
    cv.imshow('web rot ref',web_mid)
    cv.imshow('web trans ref',web_down)
    cv.waitKey(0)
    
    # exit()

    # get left frame: 
    left = 70
    fib1_rot.set(cv.CAP_PROP_POS_FRAMES, left)
    fib_left = fibrescope_process(fib1_rot.read()[1])
    web1_rot.set(cv.CAP_PROP_POS_FRAMES, left)
    web_left = webcam_process(web1_rot.read()[1])
    
    # get right frame:
    right = 120
    fib1_rot.set(cv.CAP_PROP_POS_FRAMES, right)
    fib_right = fibrescope_process(fib1_rot.read()[1])
    web1_rot.set(cv.CAP_PROP_POS_FRAMES, right)
    web_right = webcam_process(web1_rot.read()[1])
    
    # get up frame: 
    up = 60
    fib1_trans.set(cv.CAP_PROP_POS_FRAMES, up)
    fib_up = fibrescope_process(fib1_trans.read()[1])
    web1_trans.set(cv.CAP_PROP_POS_FRAMES, up)
    web_up = webcam_process(web1_trans.read()[1])

    # save raw figures: 
    
    # apply OF to all of them (ref_rot = mid, ref_trans = down):
    cv.imshow('fib left OF', OF_LK(fib_left, fib_mid, 'f')[0])
    cv.imshow('fib right OF', OF_LK(fib_right, fib_mid, 'f')[0])
    cv.imshow('fib up OF', OF_LK(fib_up, fib_down, 'f')[0])

    cv.imshow('web left OF', OF_LK(web_left, web_mid, 'w')[0])
    cv.imshow('web right OF', OF_LK(web_right, web_mid, 'w')[0])
    cv.imshow('web up OF', OF_LK(web_up, web_down, 'w')[0])

    cv.waitKey(0)
    
    # save OF figures: 
    
    # quiver plots: plt.quiver(x,y,u,v) -> u = p1[...,0], v = p1[...,1]
    fib_x, fib_y, web_x, web_y = ... 

if __name__ == '__main__':
    main()
    cv.destroyAllWindows()