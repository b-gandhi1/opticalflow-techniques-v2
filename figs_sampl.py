import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt # use this for quiver plots

# fibrescope image enhancement parameters: 
CONTRAST = 3
BRIGHTNESS = 5
fib_rot_centre = (355,325)
fib_trans_centre = (420,300)

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
    img, mask_OF = None, None
        
    # LK OF parameters: 
    if img_process == 'w':
        print('LK: Webcam')
        # img_process_func = webcam_process
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
        # img_process_func = fibrescope_process
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
    
    # if img_process == 'w':
    #     frame_filt = webcam_process(frame)
    # elif img_process == 'f':
    #     frame_filt = fibrescope_process(frame, fib_rot_centre)
    # frame_filt = img_process_func(frame) # was: (cap,frame)
    # cv.imshow('FILTERED + CROPPED',frame_filt)
    frame_filt = frame # no filtering needed again, as already filtered! 
    # p1,st,err = cv.calcOpticalFlowPyrLK(ref_frame, frame_filt, p0, None, None, None,**lk_params)
    p1,st,err = cv.calcOpticalFlowPyrLK(ref_frame, frame_filt, p0, p1, st, err,**lk_params)
    magnitude, angle = cv.cartToPolar(p1[..., 0], p1[..., 1])
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # mask_OF = cv.line(mask_OF, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        mask_OF = cv.arrowedLine(mask_OF, (int(c), int(d)), (int(a), int(b)), (255,255,255), 2, tipLength=0.5)
        # frame_filt = cv.circle(frame_filt, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame_filt, mask_OF)
    # cv.imshow('Optical Flow - Lucas-Kanade', img)        
    return img, p1[..., 0], p1[..., 1]

def LK_vid(cap, ref_frame,img_process):

    cap.set(cv.CAP_PROP_POS_FRAMES, 0) # reset to frame 0
    
    # LK OF parameters: 
    if img_process == 'w':
        print('LK: Webcam')
        # img_process_func = webcam_process
        feature_params = dict( maxCorners = 700, 
                                qualityLevel = 0.15, # between 0 and 1. Lower numbers = higher quality level. 
                                minDistance = 25.0, # distance in pixels between points being monitored. 
                                blockSize = 5,
                                useHarrisDetector = False, 
                                k = 0.04 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense. 
        lk_params = dict( winSize  = (45, 45),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        filename = 'figs_sampl/LK_webcam.mp4'
    
    elif img_process == 'f':
        print('LK: Fibrescope')
        # img_process_func = fibrescope_process
        feature_params = dict( maxCorners = 100, 
                                qualityLevel = 0.01, # between 0 and 1. Lower numbers = higher quality level. 
                                minDistance = 5.0, # distance in pixels between points being monitored. 
                                blockSize = 3,
                                useHarrisDetector = False, # Shi-Tomasi better for corner detection than Harris for fibrescope. 
                                k = 0.04 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense.
        lk_params = dict( winSize = (45, 45),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        filename = 'figs_sampl/LK_fibrescope.mp4'
    
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
    
    # video object create: 
    out = cv.VideoWriter(filename,cv.VideoWriter_fourcc(*'mp4v'),10,(ref_frame.shape[1],ref_frame.shape[0]),True)
    # if img_process == 'w':
    #     frame_filt = webcam_process(frame)
    # elif img_process == 'f':
    #     frame_filt = fibrescope_process(frame, fib_rot_centre)
    # frame_filt = img_process_func(frame) # was: (cap,frame)
    # cv.imshow('FILTERED + CROPPED',frame_filt)
    while True: 
        ret, frame = cap.read()
        if not ret: break
        
        if img_process == 'w':
            frame_filt = webcam_process(frame)
        elif img_process == 'f':
            frame_filt = fibrescope_process(frame, fib_rot_centre)
        else: print("ERROR: Invalid image process method for Lukas-Kanade.")
        # p1,st,err = cv.calcOpticalFlowPyrLK(ref_frame, frame_filt, p0, None, None, None,**lk_params)
        p1,st,err = cv.calcOpticalFlowPyrLK(ref_frame, frame_filt, p0, p1, st, err,**lk_params)
        magnitude, angle = cv.cartToPolar(p1[..., 0], p1[..., 1])
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            mask_OF = cv.arrowedLine(mask_OF, (int(c), int(d)), (int(a), int(b)), (255,255,255), 2, tipLength=0.5)
            # mask_OF = cv.line(mask_OF, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            # frame_filt = cv.circle(frame_filt, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame_filt, mask_OF)        
        img_RGB = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        cv.imshow('frame',img_RGB)
        out.write(img_RGB) # save video in RBG
        
        # clear lines for next frame:
        mask_OF = np.zeros_like(mask_OF) # clear lines for next frame
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            break
    cap.release()
    out.release()
    # cv.destroyAllWindows()
    
def main():
    fib1_trans = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/fibrescope/transTz/fibrescope1-14-Feb-2024--19-22-46.mp4')
    fib1_rot = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/fibrescope/rotation/fibrescope1-05-Feb-2024--16-54-31.mp4')
    web1_rot = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/webcam/rotation/webcam1-05-Feb-2024--15-03-50.mp4')
    web1_trans = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/webcam/transTz/webcam1-14-Feb-2024--20-01-16.mp4')

    # get mid/down frame for all: 
    ret1, fib_mid = fib1_rot.read()
    ret2, fib_down = fib1_trans.read()
    web1_rot.set(cv.CAP_PROP_POS_FRAMES, 40) 
    ret3, web_mid = web1_rot.read()
    web1_trans.set(cv.CAP_PROP_POS_FRAMES, 2)
    ret4, web_down = web1_trans.read()

    if not ret1 or not ret2 or not ret3 or not ret4: print('ERROR: Cannot get frame.')
    
    # get mid/down frame (aka reference frames): (mid for rotation, down for translation)
    fib_mid = fibrescope_process(fib_mid, fib_rot_centre)
    fib_down = fibrescope_process(fib_down, fib_trans_centre)
    web_mid = webcam_process(web_mid)
    web_down = webcam_process(web_down)
    
    # show frames 
    # cv.imshow('fib rot ref',fib_mid)
    # cv.imshow('fib trans ref',fib_down)
    # cv.imshow('web rot ref',web_mid)
    # cv.imshow('web trans ref',web_down)
    # cv.waitKey(0)
    
    # get left frame: 
    left = 70
    fib1_rot.set(cv.CAP_PROP_POS_FRAMES, left)
    fib_left = fibrescope_process(fib1_rot.read()[1], fib_rot_centre)
    web_left = 160
    web1_rot.set(cv.CAP_PROP_POS_FRAMES, web_left)
    web_left = webcam_process(web1_rot.read()[1])
    
    # get right frame:
    right = 120
    fib1_rot.set(cv.CAP_PROP_POS_FRAMES, right)
    fib_right = fibrescope_process(fib1_rot.read()[1], fib_rot_centre)
    web_right = 210
    web1_rot.set(cv.CAP_PROP_POS_FRAMES, web_right)
    web_right = webcam_process(web1_rot.read()[1])
    
    # get up frame: LK WAS NOT USED FOR TRANSLATION... 
    # up = 60
    # fib1_trans.set(cv.CAP_PROP_POS_FRAMES, up)
    # fib_up = fibrescope_process(fib1_trans.read()[1], fib_trans_centre)
    # web1_trans.set(cv.CAP_PROP_POS_FRAMES, up)
    # web_up = webcam_process(web1_trans.read()[1])

    # save raw figures: 
    
    # apply OF to all of them (ref_rot = mid, ref_trans = down):
    fib_left_img, fib_left_p1_x, fib_left_p1_y = OF_LK(fib_left, fib_mid, 'f')
    cv.imshow('fib left OF', fib_left_img)
    cv.imwrite('figs_sampl/fib_left_OF.png', fib_left_img)
    fib_right_img, fib_right_p1_x, fib_right_p1_y = OF_LK(fib_right, fib_mid, 'f')
    cv.imshow('fib right OF', fib_right_img)
    cv.imwrite('figs_sampl/fib_right_OF.png', fib_right_img)

    web_left_img, web_left_p1_x, web_left_p1_y = OF_LK(web_left, web_mid, 'w')
    cv.imshow('web left OF', web_left_img)
    cv.imwrite('figs_sampl/web_left_OF.png', web_left_img)
    web_right_img, web_right_p1_x, web_right_p1_y = OF_LK(web_right, web_mid, 'w')
    cv.imshow('web right OF', web_right_img)
    cv.imwrite('figs_sampl/web_right_OF.png', web_right_img)

    cv.waitKey(0)
    
    # call LK_vid function:
    # fib1_rot.set(cv.CAP_PROP_POS_FRAMES, 0)
    # web1_rot.set(cv.CAP_PROP_POS_FRAMES, 0)
    LK_vid(fib1_rot, fib_mid, 'f')
    LK_vid(web1_rot, web_mid, 'w')

if __name__ == '__main__':
    main()
    cv.destroyAllWindows()