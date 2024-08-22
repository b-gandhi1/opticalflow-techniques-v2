import numpy as np
import pandas as pd	
from scipy.spatial.transform import Rotation as R
# import ast
import matplotlib.pyplot as plt
import sys
import os

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
    # q0 = Q[0] # w
    # q1 = Q[1] # x
    # q2 = Q[2] # y
    # q3 = Q[3] # z
    
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
def transform_franka_pillow(w,x,y,z,euler_seq):
    # trans_mat = np.array([[-1,0,0,t_x],[0,1,0,t_y],[0,0,-1,t_z],[0,0,0,1]])
    # eeRoty43 = np.array([[-0.7314,0,0.6820],[0,1,0],[0.6820,0,0.7314]]) # 43 deress rotation around y axis of ee frame OR 137 degree rotation of base frame around y axis.
    # frame_rot_base2pillow = np.array([[0,1,0],[1,0,0],[0,0,-1]])
    # trans_mat = np.matmul(frame_rot_base2pillow,eeRoty43) # this did not work either.. :()
    # trans_mat = frame_rot_base2pillow
    orig_euler = R.from_quat([w,x,y,z]).as_euler(euler_seq, degrees=True)
    # orig_euler = euler_from_quaternion(x,y,z,w)
    # quat2rot = np.matmul(quaternion_rotation_matrix(w,x,y,z),eeRoty43) # This did not work :( Try again.. 
    # quat2rot = quaternion_rotation_matrix(w,x,y,z)
    # quat2rot = R.from_quat([w,x,y,z]).as_matrix()
    # print(quat2rot)
    # transformed = np.matmul(quat2rot,trans_mat) 
    
    # trans_euler = R.from_matrix(transformed).as_euler(euler_seq, degrees=True)
    
    # back2quat = R.from_euler('xyz', trans_euler, degrees=False).as_quat()
    
    # return trans_euler[0],trans_euler[1],trans_euler[2]
    return orig_euler[1],orig_euler[0]*(-1),orig_euler[2]*(-1) # transformations applied directly here
    # return back2quat[1], back2quat[2], back2quat[3]
# convert data
def convertdata(data,euler_seq): 
    rotations = pd.DataFrame(columns=['roll_x','pitch_y','yaw_z']) # dataframe init for saving output
    
    for _,quat in data.iterrows(): # read quaternions file > convert to euler > write to dataframe
    
        w = float(quat.loc['Franka Rw'])
        x = float(quat.loc['Franka Rx'])
        y = float(quat.loc['Franka Ry'])
        z = float(quat.loc['Franka Rz'])
        
        roll_x,pitch_y,yaw_z = transform_franka_pillow(w,x,y,z,euler_seq)

        var = pd.DataFrame({'roll_x':[roll_x],'pitch_y':[pitch_y],'yaw_z':[yaw_z]}).astype(float)
        rotations = pd.concat([rotations,var],ignore_index=True)
        
    return rotations

def plt_euler(rollX,pitchY,yawZ):
    plt.figure()
    plt.plot(rollX)
    plt.plot(pitchY)
    plt.plot(yawZ)
    plt.legend(['roll_x','pitch_y','yaw_z'])
    plt.grid()
    plt.tight_layout()
    # plt.show()
def main(path, pitchroll, i, euler_seq):
    print("PATH READ FILE --------> ",path)

    # load data > extract data
    
    # load ground truth
    raw_quat_data = pd.read_csv(path, delimiter=',',dtype={'Franka Rx': float,'Franka Ry': float,'Franka Rz': float,'Franka Rw': float})   # , skiprows=[i for i in range(13)]
    
    # run conversions and save to csv
    euler_data = convertdata(raw_quat_data,euler_seq)
    
    if pitchroll == "pitch":
        pitchroll_dir = "LK_pitch"
    elif pitchroll == "roll":
        pitchroll_dir = "LK_roll"
    
    # save as csv
    euler_data.to_csv(os.path.join('imu-fusion-outputs', pitchroll_dir, pitchroll+i+'euler_gnd.csv'), header=['roll_x','pitch_y','yaw_z'])
    
    # plots:
    plt_euler(euler_data.iloc[:,0],euler_data.iloc[:,1],euler_data.iloc[:,2])

    plt.show()
    
if __name__ == "__main__":
    path = sys.argv[1] # pitch1 or roll1
    
    length = len(path)
    pitchroll, i = path[:length-1], path[length-1] # split number at the endfrom filename 

    if pitchroll == "pitch":
        dirs = "pitch_4-jun-2024/fibrescope"+i
        euler_seq = "zxy"
    elif pitchroll == "roll":
        dirs = "roll_6-jun-2024/fibrescope"+i
        euler_seq = "zyx"
    else:
        print("ERROR: Unrecognised input for pressure selector.")
        exit()

    path_gen = "data_collection_with_franka/B07LabTrials/imu-sensor-fusion/"+dirs+".csv"
    
    main(path_gen, pitchroll, i, euler_seq) 