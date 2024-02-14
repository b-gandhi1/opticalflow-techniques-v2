import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from trial_img_process_calib import webcam_process, fibrescope_process
from linkingIO import normalize_vector, statistics_calc
brightness_z = []

WIDTH = 640
HEIGHT = 480

# fibrescope image enhancement parameters: 
CONTRAST = 3
BRIGHTNESS = 5

def fibrescope_process(frame):
    
    kernel = np.ones((2,2),np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    # x,y,w,h = 140,70,200,150 # after resizing frame size. UPDATE THIS
    # x,y,w,h = 350,280,200,110 # after resizing frame size. UPDATE THIS
    # x,y,w,h = 0,0,frame.shape[1],frame.shape[0] # no cropping, using all of it for now. 
    # rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    circle = cv.circle(mask_blank, (355,345), 100, (255,255,255), -1)
    masked = cv.bitwise_and(gray,gray,mask=circle)
    brightened = cv.addWeighted(masked, CONTRAST, np.zeros(masked.shape, masked.dtype), 0, BRIGHTNESS)     
    binary = cv.threshold(brightened,57,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    # eroded = cv.erode(binary,np.ones((4,3),np.uint8))
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return dilated
    
def webcam_process(frame):
    # frame = cv.resize(frame,(int(width/2),int(height/2)))
    kernel = np.ones((4,4),np.uint8)
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    x,y,w,h = 0,60,635,340 # (x,y) = top left params
    rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    masked = cv.bitwise_and(gray,gray,mask=rect)
    binary = cv.threshold(masked,50,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    # same threshold does not work for static image and video both. Static img thresh = 50, video thresh = 140. 
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return dilated 

def get_brightness(s,path):
    cap = cv.VideoCaptute(path)
    if not cap.isOpened(): print("ERROR: Cannot open camera/video file.")
    
    fps = cap.get(cv.CAP_PROP_FPS) # get FPS
    tot_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) # get tot frames
    time_len = tot_frames/fps
    cap.set(cv.CAP_PROP_FPS, fps)
    print('FPS: ',fps)
    print('Total frames: ',tot_frames)
    print('Time length: ',time_len)
    
    while True: 
        ret, frame = cap.read()
        if not ret: break

        cv.imshow('Raw img',frame)
    
        if s == 'w':
            output_frame = webcam_process(frame)
        elif s == 'f':
            output_frame = fibrescope_process(frame)
        else: 
            print('ERROR: Unrecognised input for image selector.')
        
        bright_val = np.mean(output_frame)
        brightness_z.append(bright_val)
        print('Brightness: ',bright_val)

        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            cap.release()
            cv.destroyAllWindows()
    
    print('Video processing complete.')    
    return brightness_z

def main():
    
    # ground truth load: 
    web1_Tz_gnd = pd.read_csv('...', delimiter=',', usecols=['Franka Tz'], dtype={'Franka Tz': float})
    web2_Tz_gnd = pd.read_csv('...', delimiter=',', usecols=['Franka Tz'], dtype={'Franka Tz': float})
    fib1_Tz_gnd = pd.read_csv('...', delimiter=',', usecols=['Franka Tz'], dtype={'Franka Tz': float})
    fib2_Tz_gnd = pd.read_csv('...', delimiter=',', usecols=['Franka Tz'], dtype={'Franka Tz': float})

    # load videos, and process them
    # web1: 
    web1_path = ...
    web1Tz = get_brightness('w',web1_path)
    
    # web2: 
    web2_path = ...
    web2Tz = get_brightness('w',web2_path)
    
    # fib1: 
    fib1_path = ...
    fib1Tz = get_brightness('f',fib1_path)
    
    # fib2: 
    fib2_path = ...
    fib2Tz = get_brightness('f',fib2_path)
        
    # statistics
    web1_stats = statistics_calc(web1Tz,web1_Tz_gnd)
    web1_df = pd.DataFrame(web1_stats,columns=['Spearman Corr (non-lin)', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)

    web2_stats = statistics_calc(web2Tz,web2_Tz_gnd)
    web2_df = pd.DataFrame(web2_stats,columns=['Spearman Corr (non-lin)', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)

    fib1_stats = statistics_calc(fib1Tz,fib1_Tz_gnd)
    fib1_df = pd.DataFrame(fib1_stats,columns=['Spearman Corr (non-lin)', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)

    fib2_stats = statistics_calc(fib2Tz,fib2_Tz_gnd)
    fib2_df = pd.DataFrame(fib2_stats,columns=['Spearman Corr (non-lin)', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)

    # merge dataframes
    merged_df = pd.concat([web1_df, web2_df, fib1_df, fib2_df], axis=0)
    
    # save as csv
    merged_df.to_csv('OF_outputs/Tz_statistics.csv', index=False)
    
if __name__ == '__main__':
    main()    
    
# get first frame from videos. 
# apply the same image processing methods
# get avg pixel brightness for this 
# get ground truth Tz values. 
# compare using spearmans correlation? if it works for scalar values too? 
