import numpy as np
import cv2 

# calibration parameters
DIM = (2464, 2056)
K = np.array([[763.6649353797775, 0.0, 1128.4822497983432], [0.0, 760.303587602483, 541.7038042528387], [0.0, 0.0, 1.0]])
D = np.array([[-0.11946056194662491], [1.4476053456229747], [-12.168935931739604], [35.90611301155222]])

# def undistort(balance = 0.7): # vary balance parameter for FOV moderation
def undistort():
    img_path = "data collection with franka/ViconLab/fibrescope/calibrationcheckboard/calib20.png"
    img = cv2.imread(img_path)
    
    # -------------------------------------------------------
    h,w = img.shape[:2]    
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    
    cv2.imshow("distorted img", img)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TO PRESERVE PREIMETER DATA, DO THIS: ------------------
    
    # dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort    
    # assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"    
    
    # if not dim2:
    #     dim2 = dim1    
    
    # if not dim3:
    #     dim3 = dim1    
        
    # scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    # scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0    
    
    # # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    
    # cv2.imshow("distorted img", img)
    # cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == '__main__':
    undistort()