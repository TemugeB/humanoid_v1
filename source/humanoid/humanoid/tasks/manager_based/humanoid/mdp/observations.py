# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg, ObservationTermCfg
from isaaclab.sensors import ContactSensor


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.envs import mdp


def base_yaw_roll(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat((yaw.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)


def base_yaw_pitch_roll(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw, pitch, and roll of the base in the simulation world frame."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Extract Euler angles (XYZ convention)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)

    # Normalize to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat((yaw.unsqueeze(-1),
                      pitch.unsqueeze(-1),
                      roll.unsqueeze(-1)), dim=-1)

def base_up_proj(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = -asset.data.projected_gravity_b

    return base_up_vec[:, 2].unsqueeze(-1)


def base_heading_proj(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projection of the base forward vector onto the world forward vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    to_target_pos[:, 2] = 0.0
    to_target_dir = math_utils.normalize(to_target_pos)
    # compute base forward vector
    heading_vec = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    # compute dot product between heading and target direction
    heading_proj = torch.bmm(heading_vec.view(env.num_envs, 1, 3), to_target_dir.view(env.num_envs, 3, 1))

    return heading_proj.view(env.num_envs, 1)


def base_angle_to_target(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle between the base forward vector and the vector to the target."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w[:, :3]
    walk_target_angle = torch.atan2(to_target_pos[:, 1], to_target_pos[:, 0])
    # compute base forward vector
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to target to [-pi, pi]
    angle_to_target = walk_target_angle - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))

    return angle_to_target.unsqueeze(-1)


def motion_phase_observation(env: ManagerBasedEnv,
                             animation_fps: float,
                             num_frames: int):
    sim_time = mdp.current_time_s(env).squeeze(-1)

    #fractional frame index
    frame_idx = sim_time * animation_fps

    # warp to trajectory length
    phase = (frame_idx % num_frames) / num_frames
    return phase.unsqueeze(-1)

class joint_acceleration(ManagerTermBase):

    def __call__(self, 
                 env, 
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

        asset: Articulation = env.scene[asset_cfg.name]

        if not hasattr(self, 'prev_vel'):
            self.prev_vel = mdp.joint_vel_rel(env, asset_cfg)
        
        #calculate the change in velocity
        current_joint_vel = mdp.joint_vel_rel(env, asset_cfg)
        accelerations = current_joint_vel - self.prev_vel
        
        #save the current vel for next call
        self.prev_vel = current_joint_vel

        return accelerations

def joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc


def feet_contact(env, sensor_cfg: SceneEntityCfg):

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    contacts = (contacts>=1.0).float()
    return contacts