import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def webcam_process(frame):

    # width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # frame = cv.resize(frame,(int(width/2),int(height/2)),)

    kernel = np.ones((4,4),np.uint8)
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    x,y,w,h = 0,60,640,340 # (x,y) = top left params
    rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    masked = cv.bitwise_and(gray,gray,mask=rect)
    binary = cv.threshold(masked,125,255,cv.THRESH_BINARY)[1] 
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return dilated 

def z_brightness(frame):
    
    norm_frame = frame/np.max(frame)
    bright_avg = np.mean(norm_frame)
    return bright_avg

def main(path):
    cap = cv.VideoCapture(path)
    if not cap.isOpened(): print("ERROR: Cannot open camera/video file.")
    
    # z_bright = None
    plt.ion()
    plt.figure('Z brightness')
    # plt.plot(z_bright,'ro')
    t=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t += 1
        frame = webcam_process(frame)
        z_bright = z_brightness(frame)
        print(z_bright)
        cv.imshow('frame',frame)
        if z_bright > 0.006:
            plt.plot(t,z_bright,'r.')
        if cv.waitKey(10) & 0xFF == ord('q'):
            print('Quitting...')
            break
        
    cap.release()
    cv.destroyAllWindows()
    plt.ioff()
    plt.show()
if __name__ == '__main__':
    path = sys.argv[1]
    main(path)