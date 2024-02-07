import numpy as np
import pandas as pd	
from scipy.spatial.transform import Rotation as R
# import ast
import matplotlib.pyplot as plt

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1)
    roll_x = np.arctan2(t0,t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    # pitch_y = math.asin(t2)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw_z = math.atan2(t3, t4)
    yaw_z = np.arctan2(t3,t4)

    return roll_x, pitch_y, yaw_z # in radians

def quaternion_rotation_matrix(q0,q1,q2,q3):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
            This rotation matrix converts a point in the local reference 
            frame to a point in the global reference frame.
    """
    # Extract the values from Q
    # q0 = Q[0]
    # q1 = Q[1]
    # q2 = Q[2]
    # q3 = Q[3]
    
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    # rot_matrix = np.array([[r00, r01, r02, 0],
    #                         [r10, r11, r12, 0],
    #                         [r20, r21, r22, 0],
    #                         [0, 0, 0, 1]])
    
    rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                            
    return rot_matrix

# transformation matrix apply
def transform_franka_pillow(w,x,y,z):
    # trans_mat = np.array([[-1,0,0,t_x],[0,1,0,t_y],[0,0,-1,t_z],[0,0,0,1]])
    trans_mat = np.array([[-1,0,0],[0,1,0],[0,0,-1],[0,0,0]]) # use later. 
    rotation = R.from_matrix(trans_mat).as_quat()
    # [x',y',z'] = Rotation_matrix * [x,y,z] --- MATRIX MULTIPLICATION RULE
    # orig_euler = R.from_quat([w,x,y,z]).as_euler('xyz', degrees=False)
    
    orig_euler = euler_from_quaternion(x,y,z,w)
    
    # apply rotation: --------
    # p, q = np.array([0,x,y,z]), rotation
    # # two step multiplication process 
    # new_p = (1/q) * p * q
    # new_p = q * new_p * (1/q)

    # trans_euler = R.from_quat(new_p).as_euler('xyz', degrees=False)
    
    return orig_euler[0],orig_euler[1],orig_euler[2]

# convert data
def convertdata(data): 
    # data = fib_gnd_truth_df1, fib_gnd_truth_df2, web_gnd_truth_df1, web_gnd_truth_df2 # one of these
    rotations = pd.DataFrame(columns=['roll_x','pitch_y','yaw_z'])
    # print('shape of data: ',np.shape(data))
    for _,quat in data.iterrows(): 
        
        # if type(quat) == str:
        #     continue
        
        # print(quat)
        # print(type(quat))
        w = float(quat.iloc[11])
        x = float(quat.iloc[8])
        y = float(quat.iloc[9])
        z = float(quat.iloc[10])
        
        # print('quat contents: ',w,x,y,z)
        # print(type(w),type(x),type(y),type(z))
        # trans_w,trans_x,trans_y,trans_z = transform_franka_pillow(w,x,y,z)
        # roll_x, pitch_y, yaw_z = euler_from_quaternion(trans_w,trans_x,trans_y,trans_z)
        roll_x,pitch_y,yaw_z = euler_from_quaternion(x,y,z,w)
        
        # rotations switch axes: 
        roll_x_new, pitch_y_new, yaw_z_new = pitch_y, roll_x, -yaw_z
        
        var = pd.DataFrame({'roll_x':[roll_x_new],'pitch_y':[pitch_y_new],'yaw_z':[yaw_z_new]}).astype(float)
        rotations = pd.concat([rotations,var],ignore_index=True)
        # print('quat contents: --')
        # print(var)
        
    # print('data type for rotations: ',type(rotations))
    return rotations

def plt_euler(rollX,pitchY,yawZ):
    plt.figure()
    plt.plot(rollX)
    plt.plot(pitchY)
    plt.plot(yawZ)
    plt.legend(['roll_x','pitch_y','yaw_z'])
    # plt.show()
def main():
    
    # load data > extract data
    # fib_df1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope1-20-Nov-2023--14-06-58.csv', delimiter=',',dtype={'Franka a': float,'Franka bi': float,'Franka cj': float,'Franka EE jk': float},header=None)
    # fib_gnd_truth_df1 = fib_df1.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions

    # fib_df2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope2-20-Nov-2023--14-09-23.csv', delimiter=',',dtype={'Franka a': float,'Franka bi': float,'Franka cj': float,'Franka EE jk': float},header=None)
    # fib_gnd_truth_df2 = fib_df2.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions

    # web_df1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam1-20-Nov-2023--15-56-11.csv', delimiter=',',dtype={'Franka a': float,'Franka bi': float,'Franka cj': float,'Franka EE jk': float},header=None)
    # web_gnd_truth_df1 = web_df1.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions

    # web_df2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam2-20-Nov-2023--15-59-11.csv', delimiter=',',dtype={'Franka a': float,'Franka bi': float,'Franka cj': float,'Franka EE jk': float},header=None)
    # web_gnd_truth_df2 = web_df2.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions

    fib1gnd = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope1-05-Feb-2024--16-55-31.csv', delimiter=',',dtype={'Franka Rx': float,'Franka Ry': float,'Franka Rz': float,'Franka Rw': float})
    fib2gnd = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope2-05-Feb-2024--17-25-47.csv', delimiter=',',dtype={'Franka Rx': float,'Franka Ry': float,'Franka Rz': float,'Franka Rw': float})
    web1gnd = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam1-05-Feb-2024--15-04-50.csv', delimiter=',',dtype={'Franka Rx': float,'Franka Ry': float,'Franka Rz': float,'Franka Rw': float})
    web2gnd = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam2-05-Feb-2024--15-15-37.csv', delimiter=',',dtype={'Franka Rx': float,'Franka Ry': float,'Franka Rz': float,'Franka Rw': float})

    
    # run conversions and save to csv
    fib1euler = convertdata(fib1gnd)
    # fib1euler.to_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib1euler.csv')    
    fib2euler = convertdata(fib2gnd)
    # fib2euler.to_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib2euler.csv')
    web1euler = convertdata(web1gnd)
    # web1euler.to_csv('data_collection_with_franka/B07LabTrials/final/webcam/web1euler.csv')
    web2euler = convertdata(web2gnd)
    # web2euler.to_csv('data_collection_with_franka/B07LabTrials/final/webcam/web2euler.csv')
    
    # print(fib1euler)
    # plots:
    plt_euler(fib1euler.iloc[:,0],fib1euler.iloc[:,1],fib1euler.iloc[:,2])
    plt_euler(fib2euler.iloc[:,0],fib2euler.iloc[:,1],fib2euler.iloc[:,2])
    plt_euler(web1euler.iloc[:,0],web1euler.iloc[:,1],web1euler.iloc[:,2])
    plt_euler(web2euler.iloc[:,0],web2euler.iloc[:,1],web2euler.iloc[:,2])
    plt.show()
if __name__ == "__main__":
    main() 