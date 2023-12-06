import cv2 as cv
import numpy as np
import susan_corner_detector as susan

# fibrescope image enhancement parameters: 
CONTRAST = 3
BRIGHTNESS = 5

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

def fibrescope_process(frame):

    # width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # frame = cv.resize(frame,(int(width/2),int(height/2)),)

    kernel = np.ones((2,2),np.uint8)
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    mask_blank = np.zeros_like(gray,dtype='uint8') # ,dtype='uint8'
    # x,y,w,h = 350,280,200,110 # after resizing frame size. 
    # rect = cv.rectangle(mask_blank, (x, y), (x+w, y+h), (255,255,255), -1) # mask apply
    circle = cv.circle(mask_blank, (430,215), 100, (255,255,255), -1)
    masked = cv.bitwise_and(gray,gray,mask=circle)
    brightened = cv.addWeighted(masked, CONTRAST, np.zeros(masked.shape, masked.dtype), 0, BRIGHTNESS)     
    binary = cv.threshold(brightened,55,255,cv.THRESH_BINARY)[1] # might remove: + cv.thresh_otsu
    morph_open = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    morph_close = cv.morphologyEx(morph_open,cv.MORPH_CLOSE,kernel)
    dilated = cv.dilate(morph_close,kernel)

    return dilated

# Shi-Tomasi Corner Detection ---------------------------------------------------
def shiTomasi(fibrescope,webcam):    
    # gray_fib = cv.cvtColor(fibrescope,cv.COLOR_BGR2GRAY) # Converting to grayscale
    # gray_web = cv.cvtColor(webcam,cv.COLOR_BGR2GRAY)
    
    fib_features = dict( maxCorners = 100, # 100 max val, and works best
                            qualityLevel = 0.01, # between 0 and 1. Lower numbers = higher quality level. 
                            minDistance = 5.0, # distance in pixels between points being monitored. 
                            blockSize = 3,
                            useHarrisDetector = False, # Shi-Tomasi better for corner detection than Harris for fibrescope. 
                            k = 0.04 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense.
    web_features = dict( maxCorners = 1000, # 100 max val, and works best
                            qualityLevel = 0.01, # between 0 and 1. Lower numbers = higher quality level. 
                            minDistance = 20.0, # distance in pixels between points being monitored. 
                            blockSize = 3,
                            useHarrisDetector = False, 
                            k = 0.04 ) # something to do with area density, starts centrally. high values spread it out. low values keep it dense. 

    corners_fib = cv.goodFeaturesToTrack(fibrescope, mask = None, **fib_features) # Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 10 is the minimum euclidean distance between two corners

    corners_web = cv.goodFeaturesToTrack(webcam,mask = None, **web_features)
    
    corners_fib = np.intp(corners_fib)
    corners_web = np.intp(corners_web)
    
    for corners in corners_fib:
        
        x,y = corners.ravel()
        cv.circle(fibrescope,(x,y),3,[0,255,0],-1) # Circling the corners in green
    
    for corners in corners_web:
        
        x,y = corners.ravel()
        cv.circle(webcam,(x,y),3,[0,255,0],-1)

    cv.imshow('Fibrescope via Shi-Tomasi Corner Detection',fibrescope)
    cv.imshow('Webcam via Shi-Tomasi Corner Detection',webcam)
    
# Harris Corner Detection -------------------------------------------------------
def harris(fibrescope,webcam):
    # gray_fib = cv.cvtColor(fibrescope,cv.COLOR_BGR2GRAY) # Converting the image to grayscale
    # gray_web = cv.cvtColor(webcam,cv.COLOR_BGR2GRAY)
    
    gray_fib = np.float32(fibrescope) # Conversion to float is a prerequisite for the algorithm
    gray_web = np.float32(webcam)

    corners_fib = cv.cornerHarris(gray_fib,3,3,0.04) # 3 is the size of the neighborhood considered, aperture parameter = 3
    # k = 0.04 used to calculate the window score (R)
    corners_web = cv.cornerHarris(gray_web,3,3,0.04)
    
    fibrescope[corners_fib>0.001*corners_fib.max()] = [255] # Marking the corners in Green. 3 channel = [0,255,0], binary = [255]
    webcam[corners_web>0.001*corners_web.max()] = [255] # 3 channel = [0,255,0], binary = [255]
    
    cv.imshow('Fibrescope via Harris Corner Detection',fibrescope)
    cv.imshow('Webcam via Harris Corner Detection',webcam)
    
def main():
    # load images from video
    fibrescope_vid = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope1-20-Nov-2023--14-05-58.mp4')
    webcam_vid = cv.VideoCapture('data_collection_with_franka/B07LabTrials/final/webcam/webcam1-20-Nov-2023--15-55-11.mp4')

    # get first frame: 
    _, fibrescope = fibrescope_vid.read()
    _, webcam = webcam_vid.read()
    
    # apply filters
    fibrescope = fibrescope_process(fibrescope)
    webcam = webcam_process(webcam)
    
    # apply corner detectors 
    shiTomasi(fibrescope,webcam)
    harris(fibrescope,webcam)
    # # susan corner detection: 
    # susan_fib = susan.susan_corner_detection(fibrescope)
    # susan_fib_out = cv.cvtColor(fibrescope, cv.COLOR_GRAY2RGB)
    # susan_fib_out[susan_fib != 0] = [255, 255, 0]
    # susan_web = susan.susan_corner_detection(webcam)
    # susan_web_out = cv.cvtColor(webcam, cv.COLOR_GRAY2RGB)
    # susan_web_out[susan_web != 0] = [255, 255, 0]
    # cv.imshow('Susan corner detection, fibrescope',susan.susan_corner_detection(fibrescope))
    # cv.imshow('Susan corner detection, webcam',susan.susan_corner_detection(webcam))
    
    # release
    cv.waitKey(0)
    webcam_vid.release()
    fibrescope_vid.release()

if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
# the parameters here work pretty well for shi-tomasi and harris corner detection methods. Shi-tomasi is better.. 
# binary frames for webcam - both detectors fail. why??? 