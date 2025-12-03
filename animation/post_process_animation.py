import pickle
import numpy as np
from yourdfpy import URDF
from functools import partial
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import argparse
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

parser = argparse.ArgumentParser(description="IK solver demo")

# Positional (Required)
parser.add_argument('robot_joint_data', type=str, help='Path to robot joint rotation data.')

# Optional Flags
parser.add_argument('--urdf_path', type=str, default='../robot_model/robot.urdf')
args = parser.parse_args()

#load the urdf file to show for animation
urdf_path = args.urdf_path
urdf_model = URDF.load(urdf_path)

#open the joint rotations data
with open(args.robot_joint_data, "rb") as f:
    joint_rotations = pickle.load(f)

print(joint_rotations.keys())
print(f'Number of joint position frames: {len(joint_rotations[list(joint_rotations.keys())[0]])}')

#Apply joint rotation smoothing
def apply_smoothing(joint_rotations, alpha = 0.8):

    smoothed_rotations = {}
    for joint_name, joint_traj in joint_rotations.items():
        smoothed_traj = joint_traj.copy()
        #add the first element as last
        smoothed_traj = np.append(smoothed_traj, smoothed_traj[0])
        #apply exp smoothing
        for t in range(1, joint_traj.shape[0]):
            smoothed_traj[t] = alpha * joint_traj[t] + (1 - alpha) * smoothed_traj[t-1]
        smoothed_rotations[joint_name] = smoothed_traj[:-1]
    
    return smoothed_rotations

def make_loopable(traj, blend_frames=3, smooth_window=3, polyorder=3):
    """
    Smoothly blends the start and end of a 1D trajectory so it loops.
    
    traj: numpy array shape [T]
    blend_frames: number of frames to blend from both ends
    smooth_window: window for Savitzky-Golay smoothing (must be odd)
    polyorder: polynomial order for Savitzky-Golay
    """

    T = len(traj)
    assert blend_frames < T // 2

    traj = traj.copy()
    
    # --- 1. Compute a smooth blend weight from 0 → 1
    w = np.linspace(0, 1, blend_frames)

    # --- 2. Blend the head and tail
    # tail → head
    for i in range(blend_frames):
        traj[i] = (1 - w[i]) * traj[i] + w[i] * traj[T - blend_frames + i]
        traj[T - blend_frames + i] = (1 - w[i]) * traj[T - blend_frames + i] + w[i] * traj[i]

    # --- 3. Final smoothing with wraparound
    # pad circularly to preserve continuity
    pad = smooth_window // 2
    padded = np.concatenate([traj[-pad:], traj, traj[:pad]])
    smoothed = savgol_filter(padded, window_length=smooth_window, polyorder=polyorder)
    smoothed = smoothed[pad:-pad]

    return smoothed

def upsample_traj(traj, factor=3):
    """
    Upsample a 1D trajectory by an integer factor using monotonic Hermite (PCHIP).
    
    traj: numpy array shape [T]
    factor: integer upsampling factor
    """
    T = len(traj)
    x_old = np.arange(T)
    x_new = np.linspace(0, T - 1, T * factor)

    interpolator = PchipInterpolator(x_old, traj)
    return interpolator(x_new)

def blend_for_looping(joint_rotations):
    blended_rotations = {}
    for joint_name, joint_traj in joint_rotations.items():
        blended_rotations[joint_name] = make_loopable(joint_traj, 
                                                    blend_frames=3,
                                                    smooth_window=33)

    return blended_rotations

#apply exp smoothing
smoothed_rotations = apply_smoothing(joint_rotations, alpha=0.9)

#upsample then blend
upsampled_rotations = {}
for joint_name, joint_traj in joint_rotations.items():
    upsampled_rotations[joint_name] = upsample_traj(joint_traj, factor=3)
blended_rotations = blend_for_looping(upsampled_rotations)

joint_ = 'hip_left_z'
plt.plot(joint_rotations[joint_], label = 'raw')
plt.plot(blended_rotations[joint_][::3], label = 'smoothed + blended')

plt.legend()
plt.show()

blended_anim = {key: traj[::3] for key, traj in blended_rotations.items()}

anim_config = {'frame': 0, 'sleep_duration': 0.033, 'frame_max': len(blended_anim['hip_left_y'])}
def anim_callback(scene, urdf_model, trajectory, anim_config):
    if anim_config['frame'] >= anim_config['frame_max']: anim_config['frame'] = 0

    cfg = {j: trajectory[j][anim_config['frame']] for j in trajectory.keys()}
    urdf_model.update_cfg(cfg)
    anim_config['frame'] = anim_config['frame'] + 1
    time.sleep(anim_config['sleep_duration'])

urdf_model.show(
#        callback=partial(anim_callback, urdf_model=urdf_model, trajectory = joint_rotations, anim_config=anim_config)
        callback=partial(anim_callback, urdf_model=urdf_model, trajectory = blended_anim, anim_config=anim_config)

)

print(len(blended_rotations[list(blended_rotations.keys())[0]]))
with open("postprocessed_joint_angles.pkl", "wb") as f:
    pickle.dump(blended_rotations, f)