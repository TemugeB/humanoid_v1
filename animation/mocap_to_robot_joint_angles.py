import pickle
import numpy as np
from yourdfpy import URDF
from functools import partial
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import argparse

parser = argparse.ArgumentParser(description="IK solver demo")

# Positional (Required)
parser.add_argument('mocap_data', type=str, help='Path to mocap data.')

# Optional Flags
parser.add_argument('--start_frame', type=int, default=0, help='Start frame (default: 0).')
parser.add_argument('--end_frame', type=int, default=-1, help='End frame.')
parser.add_argument('--urdf_path', type=str, default='../robot_model/robot.urdf')
args = parser.parse_args()


urdf_path = args.urdf_path
urdf_model = URDF.load(urdf_path)

#keys: robot joint names. values: mocap data entry names
joint_to_mocap_dataname = {
    'hip_left_y': 'left_waist',
    'hip_left_x': 'left_waist',
    'hip_left_z': 'left_waist',
    'hip_right_y': 'right_waist',
    'hip_right_x': 'right_waist',
    'hip_right_z': 'right_waist',
    'knee_left': 'left_knee',
    'knee_right': 'right_knee',
    'spine': 'spine',
    'left_shoulder_x': 'left_shoulder',
    'left_shoulder_y': 'left_shoulder',
    'left_shoulder_z': 'left_shoulder',
    'right_shoulder_x': 'right_shoulder',
    'right_shoulder_y': 'right_shoulder',
    'right_shoulder_z': 'right_shoulder',
    'left_elbow': 'left_elbow',
    'right_elbow': 'right_elbow'
}


angles_decomposion_order = {
    'hip_left_y': 'yxz',
    'hip_left_x': 'yxz',
    'hip_left_z': 'yxz',
    'hip_right_y': 'yxz',
    'hip_right_x': 'yxz',
    'hip_right_z': 'yxz',
    'knee_left': 'yxz',
    'knee_right': 'yxz',
    'spine': 'yxz',
    'left_shoulder_x': 'xyz',
    'left_shoulder_y': 'xyz',
    'left_shoulder_z': 'xyz',
    'right_shoulder_x': 'xyz',
    'right_shoulder_y': 'xyz',
    'right_shoulder_z': 'xyz',
    'left_elbow': 'xyz',
    'right_elbow': 'xyz'
}

#which axis does the robot joint rotation axis correspond to in the data
joint_axis_to_mocap_data_axis = {
    'hip_left_x': [0, 1, 0],
    'hip_left_y': [-1, 0, 0],
    'hip_left_z': [0, 0, 1],
    'knee_left': [-1, 0, 0],
    'hip_right_x': [0, 1, 0],
    'hip_right_y': [1, 0, 0],
    'hip_right_z': [0, 0, 1],
    'knee_right': [1, 0, 0],
    'left_shoulder_x': [-1, 0, 0],
    'left_shoulder_y': [0, -1, 0], 
    'left_shoulder_z': [0, 0, 1],
    'right_shoulder_x': [-1, 0, 1],
    'right_shoulder_y': [0, -1, 0], 
    'right_shoulder_z': [0, 0, 1],
    'left_elbow': [0, -1, 0],
    'right_elbow': [0, 1, 0],
    'spine': [0, 0, 1]
}

#Mocap is defined from T pose. Robot is not. So need an extra rotation before applying the joint rotations
robot_pose_correction = {
    'hip_left_x': 0.0,
    'hip_left_y': 0.0,
    'hip_left_z': 0.0,
    'knee_left': 0.0,
    'hip_right_x': 0.0,
    'hip_right_y': 0.0,
    'hip_right_z': 0.0,
    'knee_right': 0.0,
    'left_shoulder_x': 0.0,
    'left_shoulder_y': 0.0, 
    'left_shoulder_z': 0.0,
    'right_shoulder_x': 0.0,
    'right_shoulder_y': 0.0, 
    'right_shoulder_z': 0.0,
    'left_elbow': 0.0,
    'right_elbow': 0.0,
    'spine': 0.0
}

with open(args.mocap_data, "rb") as f:
    mocap = pickle.load(f)

print(mocap.keys())
print(f'Number of mocap frames: {len(mocap[list(mocap.keys())[0]])}')

# Mocap: quaternion
def get_joint_angles_for_all_frames(joint_name, mocap):
    """
    Returns an array of shape [num_frames] for this joint
    """
    mocap_quat = np.array(mocap[joint_to_mocap_dataname[joint_name]])[args.start_frame:args.end_frame]  # shape [num_frames, 3]
    mocap_angles_xyz = Rotation.from_quat(mocap_quat).as_euler(angles_decomposion_order[joint_name], degrees=False)

    axis_map = joint_axis_to_mocap_data_axis[joint_name]
    axis_idx = np.argmax(np.abs(axis_map))
    sign = np.sign(axis_map[axis_idx])

    # Extract and sign-correct
    angles = sign * mocap_angles_xyz[:, axis_idx]

    # Add per-joint pose correction
    angles += robot_pose_correction[joint_name]

    return angles  # shape [num_frames]


# Iterate over all robot joints
def get_all_robot_joint_angles_all_frames(mocap):
    """
    Returns a dict: robot_joint_name -> array of shape [num_frames]
    """
    robot_angles = {}
    for joint_name in joint_to_mocap_dataname.keys():
        robot_angles[joint_name] = get_joint_angles_for_all_frames(joint_name, mocap)
    return robot_angles

robot_angles = get_all_robot_joint_angles_all_frames(mocap)
# for key in robot_angles.keys():
#     plt.plot(robot_angles[key], label = key)

# plt.plot(robot_angles['hip_left_x'], label = 'x')
# plt.plot(robot_angles['hip_left_y'], label = 'y')
# plt.plot(robot_angles['hip_left_z'], label = 'z')
# plt.legend()
# plt.show()


anim_config = {'frame': 0, 'sleep_duration': 0.033, 'frame_max': len(robot_angles['hip_left_y'])}
def anim_callback(scene, urdf_model, trajectory, anim_config):
    if anim_config['frame'] >= anim_config['frame_max']: anim_config['frame'] = 0

    cfg = {j: trajectory[j][anim_config['frame']] for j in trajectory.keys()}
    urdf_model.update_cfg(cfg)
    anim_config['frame'] = anim_config['frame'] + 1
    time.sleep(anim_config['sleep_duration'])

urdf_model.show(
        callback=partial(anim_callback, urdf_model=urdf_model, trajectory = robot_angles, anim_config=anim_config)
)

cropped_mocap = {
    key: arr[args.start_frame:args.end_frame] 
    for key, arr in mocap.items()
} 

print('Generated robot angles for: ', robot_angles.keys())
# with open("robot_angles.pkl", "wb") as f:
#     pickle.dump(robot_angles, f)
