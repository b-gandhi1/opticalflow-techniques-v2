import cv2 as cv
import numpy as np

# load images from video
fibrescope_vid = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope1-20-Nov-2023--14-05-58.mp4')
webcam_vid = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/webcam/webcam1-20-Nov-2023--15-55-11.mp4')

# get first frame: 
_, fibrescope = fibrescope_vid.read()
_, webcam = webcam_vid.read()

# Shi-Tomasi Corner Detection ---------------------------------------------------
def shiTomasi(fibrescope,webcam):    
    gray_fib = cv.cvtColor(fibrescope,cv.COLOR_BGR2GRAY) # Converting to grayscale
    gray_web = cv.cvtColor(webcam,cv.COLOR_BGR2GRAY)
    
    corners_fib = cv.goodFeaturesToTrack(gray_fib,1000,0.01,10) #Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 10 is the minimum euclidean distance between two corners

    corners_web = cv.goodFeaturesToTrack(gray_web,1000,0.01,10)
    
    corners_fib = np.int0(corners_fib)
    corners_web = np.int0(corners_web)
    
    for corners in corners_fib:
        
        x,y = corners.ravel()
        cv.circle(fibrescope,(x,y),3,[0,255,0],-1) # Circling the corners in green
    
    for corners in corners_web:
        
        x,y = corners.ravel()
        cv.circle(webcam,(x,y),3,[0,255,0],-1)

    cv.imshow('Fibrescope via Shi-Tomasi Corner Detection',fibrescope)
    cv.imshow('Webcam via Shi-Tomasi Corner Detection',webcam)
    cv.waitKey(0)
    
# Harris Corner Detection -------------------------------------------------------
def harris(fibrescope,webcam):
    gray_fib = cv.cvtColor(fibrescope,cv.COLOR_BGR2GRAY) # Converting the image to grayscale
    gray_web = cv.cvtColor(webcam,cv.COLOR_BGR2GRAY)
    
    gray_fib = np.float32(gray_fib) # Conversion to float is a prerequisite for the algorithm
    gray_web = np.float32(gray_web)
    
    corners_fib = cv.cornerHarris(gray_fib,3,3,0.04) # 3 is the size of the neighborhood considered, aperture parameter = 3
    # k = 0.04 used to calculate the window score (R)
    corners_web = cv.cornerHarris(gray_web,3,3,0.04)
    
    fibrescope[corners_fib>0.001*corners_fib.max()] = [0,255,0] # Marking the corners in Green
    webcam[corners_web>0.001*corners_web.max()] = [0,255,0]
    
    cv.imshow('Fibrescope via Harris Corner Detection',fibrescope)
    cv.imshow('Webcam via Harris Corner Detection',webcam)
    cv.waitKey(0)
    
def main():
    shiTomasi(fibrescope,webcam)
    harris(fibrescope,webcam)

if __name__ == '__main__':
    main()
    cv.release()
    cv.destroyAllWindows()

# the parameters here work pretty well for shi-tomasi and harris corner detection methods. Shi-tomasi is better.. 
# why did it not work before? try that again... 
