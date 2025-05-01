from filterpy.kalman import KalmanFilter
import numpy as np

kf = KalmanFilter(dim_x=4, dim_z=2, dim_u=2)

kf.F = np.eye(4) # initial state transition matrix
kf.H = np.array([[1.,0.]]) # measurement function
kf.R = np.array([[0.1]]) # measurement noise covariance matrix
kf.Q = np.eye(2) # process noise covariance matrix

kf.x = np.zeros((4,1)) # initial state estimate
kf.P = np.eye(4) # initial state covariance matrix

def fx(x, dt):
# state transition function - predict next state based
# on constant velocity model x = vt + x_0
    F = np.array([[1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]], dtype=float)
    return np.dot(F, x)
def gu(u):
    G = np.array()
    return np.dot(G, u)
def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]
    H = np.array([[0, -1, 0, 0],
                [-1, 0, 0, 0]], dtype=float)
    return np.dot(H, x)