import numpy as np
import pandas as pd	
# import math

# load data > extract data
fib_df1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope1-20-Nov-2023--14-06-58.csv', delimiter=',')
fib_gnd_truth_df1 = fib_df1.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions

fib_df2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fibrescope2-20-Nov-2023--14-09-23.csv', delimiter=',')
fib_gnd_truth_df2 = fib_df2.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions

web_df1 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam1-20-Nov-2023--15-56-11.csv', delimiter=',')
web_gnd_truth_df1 = web_df1.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions

web_df2 = pd.read_csv('data_collection_with_franka/B07LabTrials/final/webcam/webcam2-20-Nov-2023--15-59-11.csv', delimiter=',')
web_gnd_truth_df2 = web_df2.iloc[1:,5:] # remove first data point to match sizes, and extract quaternions

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

# convert data
def convertdata(data): 
    # data = fib_gnd_truth_df1, fib_gnd_truth_df2, web_gnd_truth_df1, web_gnd_truth_df2 # one of these
    
    rotations = pd.DataFrame(columns=['roll_x','pitch_y','yaw_z'])
    
    for quat in data: 
        
        if type(quat) == str:
            continue
        
        print(quat)
        print(type(quat))
        w = data.iloc(quat)[0]
        x = data.iloc(quat)[1]
        y = data.iloc(quat)[2]
        z = data.iloc(quat)[3]
        
        print('quat contents: ',quat)
        roll_x, pitch_y, yaw_z = euler_from_quaternion(x, y, z, w)
        
        var = pd.DataFrame({'roll_x':[roll_x],'pitch_y':[pitch_y],'yaw_z':[yaw_z]})
        rotations = pd.concat([rotations,var],ignore_index=True)
        print('quat contents: --')
        print(var)
        
        print('data type for rotations: ',type(rotations))
        return rotations

def main():
    fib1euler = convertdata(fib_gnd_truth_df1)
    fib1euler.to_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib1euler.csv')    
    fib2euler = convertdata(fib_gnd_truth_df2)
    fib2euler.to_csv('data_collection_with_franka/B07LabTrials/final/fibrescope/fib2euler.csv')
    web1euler = convertdata(web_gnd_truth_df1)
    web1euler.to_csv('data_collection_with_franka/B07LabTrials/final/webcam/web1euler.csv')
    web2euler = convertdata(web_gnd_truth_df2)
    web2euler.to_csv('data_collection_with_franka/B07LabTrials/final/webcam/web2euler.csv')
    
if __name__ == "__main__":
    main()