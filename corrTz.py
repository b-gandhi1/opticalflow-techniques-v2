import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trial_img_process_calib import webcam_process, fibrescope_process
from linkingIO import normalize_vector, statistics_calc
brightness_z = []
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
