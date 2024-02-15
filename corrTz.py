import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from trial_img_process_calib import webcam_process, fibrescope_process
from linkingIO import normalize_vector #, statistics_calc
from scipy.stats import pearsonr, spearmanr, weightedtau, kendalltau

WIDTH = 640
HEIGHT = 480

# fibrescope image enhancement parameters: 
CONTRAST = 3
BRIGHTNESS = 5

def fibrescope_process(frame):
    
    kernel = np.ones((2,2),np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    circle = cv.circle(mask_blank, (420,300), 100, (255,255,255), -1)
    masked = cv.bitwise_and(gray,gray,mask=circle)
    brightened = cv.addWeighted(masked, CONTRAST, np.zeros(masked.shape, masked.dtype), 0, BRIGHTNESS)     
    binary = cv.threshold(brightened,55,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    # eroded = cv.erode(binary,np.ones((4,3),np.uint8))
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return brightened
    
def webcam_process(frame):
    kernel = np.ones((4,4),np.uint8)
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    x,y,w,h = 0,60,635,340 # (x,y) = top left params
    rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    masked = cv.bitwise_and(gray,gray,mask=rect)
    binary = cv.threshold(masked,231,255,cv.THRESH_BINARY)[1] 
    # same threshold does not work for static image and video both. Static img thresh = 50, video thresh = 140. 
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return masked 

def get_brightness(s,path):
    brightness_z = [] # init empty
    cap = cv.VideoCapture(path)
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

        # cv.imshow('Raw img',frame)
    
        if s == 'w':
            output_frame = webcam_process(frame)
        elif s == 'f':
            output_frame = fibrescope_process(frame)
        else: 
            print('ERROR: Unrecognised input for image selector.')
        
        cv.imshow('Processed img',output_frame)
        bright_val = np.mean(output_frame)
        brightness_z.append(bright_val)
        print('Brightness: ',bright_val)

        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            cap.release()
            cv.destroyAllWindows()
    
    print('Video processing complete.')    
    return brightness_z

def statistics_calc(experimental_data, ground_truth):
    # corr_linear, _ = pearsonr(experimental_data, ground_truth,alternative='two-sided') # pearson correlation (linear). Low corr ~= 0. -0.5 < poor corr < 0.5. -1 < corr_range < 1. 
    # variables can be +vely or -vely linearly correlated. 
    corr_nonlin, _ = spearmanr(experimental_data, ground_truth,alternative='two-sided',nan_policy='propagate') # spearman correlation (nonlinear). High corr ~= 1. -1 < corr_range < 1. same as pearson.
    # r_sq = metrics.r2_score(ground_truth,experimental_data,force_finite=False)
    corr_kendalltau = kendalltau(ground_truth,experimental_data)
    corr_weightedtau = weightedtau(ground_truth,experimental_data, rank=True, weigher=None, additive=False)
    
    # print(corr_nonlin, corr_kendalltau[0], corr_kendalltau[1], corr_weightedtau[0])
    # TEST THE COMMANDS ABOVE !!!! ----------
    # return corr_linear, corr_nonlin, r_sq, corr_kendalltau[0], corr_kendalltau[1], corr_weightedtau[0]
    return np.array([corr_nonlin, corr_kendalltau[0], corr_kendalltau[1], corr_weightedtau[0]])

def main():
    
    # ground truth load: 
    web1_Tz_gnd = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/transTz/webcam1-14-Feb-2024--20-02-16.csv', delimiter=',', usecols=['Franka Tz'], dtype={'Franka Tz': float})
    web1_Tz_gnd = web1_Tz_gnd.iloc[10:int(0.5*len(web1_Tz_gnd)),:]*(-1) # trim half, add 10 to remove zeros
    web2_Tz_gnd = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/transTz/webcam2-14-Feb-2024--20-04-07.csv', delimiter=',', usecols=['Franka Tz'], dtype={'Franka Tz': float})
    web2_Tz_gnd = web2_Tz_gnd.iloc[10:int(0.5*len(web2_Tz_gnd)),:]*(-1) # trim half
    fib1_Tz_gnd = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/transTz/fibrescope1-14-Feb-2024--19-23-47.csv', delimiter=',', usecols=['Franka Tz'], dtype={'Franka Tz': float})
    fib1_Tz_gnd = fib1_Tz_gnd.iloc[10:int(0.5*len(fib1_Tz_gnd)),:]*(-1)
    fib2_Tz_gnd = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/transTz/fibrescope2-14-Feb-2024--19-26-32.csv', delimiter=',', usecols=['Franka Tz'], dtype={'Franka Tz': float})
    fib2_Tz_gnd = fib2_Tz_gnd.iloc[10+int(0.5*len(fib2_Tz_gnd)):int(len(fib2_Tz_gnd)),:]*(-1)
    # check sizes:
    # print('Gnd truth sizes: ',len(web1_Tz_gnd),len(web2_Tz_gnd),len(fib1_Tz_gnd),len(fib2_Tz_gnd))
    # load videos, and process them
    # web1: 
    web1_path = 'data_collection_with_franka/B07LabTrials/final/webcam/transTz/webcam1-14-Feb-2024--20-01-16.mp4'
    web1Tz = get_brightness('w',web1_path)
    web1Tz = web1Tz[10:int(0.5*len(web1Tz))]
    
    # web2: 
    web2_path = 'data_collection_with_franka/B07LabTrials/final/webcam/transTz/webcam2-14-Feb-2024--20-03-07.mp4'
    web2Tz = get_brightness('w',web2_path)
    web2Tz = web2Tz[10:int(0.5*len(web2Tz))]
    
    # fib1: 
    fib1_path = 'data_collection_with_franka/B07LabTrials/final/fibrescope/transTz/fibrescope1-14-Feb-2024--19-22-46.mp4'
    fib1Tz = get_brightness('f',fib1_path)
    fib1Tz = fib1Tz[10:int(0.5*len(fib1Tz))]
    # fib2: 
    fib2_path = 'data_collection_with_franka/B07LabTrials/final/fibrescope/transTz/fibrescope2-14-Feb-2024--19-25-32.mp4'
    fib2Tz = get_brightness('f',fib2_path)
    fib2Tz = fib2Tz[10+int(0.5*len(fib2Tz)):int(len(fib2Tz))]
    
    # check sizes:
    # print('Bright sizes: ',len(web1Tz),len(web2Tz),len(fib1Tz),len(fib2Tz))
    # plot: 
    t = np.linspace(0,30,len(web1Tz))
    print('t size',len(t))
    plt.figure()
    plt.plot(t,normalize_vector(web1Tz))
    plt.plot(t,normalize_vector(web2Tz))
    plt.plot(t,normalize_vector(fib1Tz))
    plt.plot(t,normalize_vector(fib2Tz))
    plt.plot(t,normalize_vector(web1_Tz_gnd))
    plt.legend(['web1','web2','fib1','fib2','GND truth'])
    plt.tight_layout()
    plt.show()
    
    # statistics
    
    web1_stats = statistics_calc(normalize_vector(web1Tz),normalize_vector(web1_Tz_gnd))
    # print(np.shape(web1_stats)) # print size of web1_stats    
    # web1_df = pd.DataFrame(web1_stats[0],web1_stats[1],web1_stats[2],web1_stats[3],columns=['Spearman Corr (non-lin)', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)

    web2_stats = statistics_calc(normalize_vector(web2Tz),normalize_vector(web2_Tz_gnd))
    # web2_df = pd.DataFrame(web2_stats,columns=['Spearman Corr (non-lin)', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)

    fib1_stats = statistics_calc(normalize_vector(fib1Tz),normalize_vector(fib1_Tz_gnd))
    # fib1_df = pd.DataFrame(fib1_stats,columns=['Spearman Corr (non-lin)', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)

    fib2_stats = statistics_calc(normalize_vector(fib2Tz),normalize_vector(fib2_Tz_gnd))
    # fib2_df = pd.DataFrame(fib2_stats,columns=['Spearman Corr (non-lin)', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)

    stats_inp = np.vstack((web1_stats,web2_stats,fib1_stats,fib2_stats,np.zeros(4)))
    # merge dataframes
    stats_Tz = pd.DataFrame(stats_inp,columns=['Spearman Corr (non-lin)', 'Kendall Tau', 'kendall p-val', 'Weighted Tau'],dtype=float)

    # merged_df = pd.concat([stats_Tz,stats_inp, 0], axis=0)
    
    # save as csv
    stats_Tz.to_csv('OF_outputs/Tz_statistics.csv', index=False)
    
if __name__ == '__main__':
    main()    
