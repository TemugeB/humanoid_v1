# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from isaaclab.envs import mdp
from isaaclab.utils.math import quat_conjugate, quat_apply, quat_mul, wrap_to_pi
from . import observations as obs
import pickle


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)
    

class joint_angle_tracking(ManagerTermBase):

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):

        # open the trajectory data and convert to torch tensor
        traj_path = '/home/temuge/robots/animations/joint_trajectory.pkl'
        traj_path = '/home/temuge/my_bots/joint_trajectory.pkl'
        with open(traj_path, "rb") as f:
            joint_traj = pickle.load(f)
        self.joint_traj = {k: torch.tensor(v, device=env.device) for k,v in joint_traj.items()}

    def __call__(self, 
                 env: ManagerBasedRLEnv, 
                 animation_fps: float, 
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset : Articulation = env.scene[asset_cfg.name]

        #Get the current time of each environment
        sampling_times = mdp.current_time_s(env).squeeze(-1)

        #For each limb, calculate the reward.
        # TODO: vectorize
        rewards = []
        for joint_name, curve in self.joint_traj.items():
            num_frames = len(curve)

            # fractional index
            frame_idx = sampling_times * animation_fps

            # wrap animation
            frame_idx = frame_idx % num_frames

            # get the frames to the left and right
            i0 = frame_idx.floor().long()
            i1 = (i0 + 1) % num_frames

            # interpolation weight
            weights = frame_idx - i0

            # linear interpolate
            joint_targets = (1 - weights) * curve[i0] + weights * curve[i1]

            # get the current joint positions
            joint_index = asset.find_joints([joint_name])[0]
            #current_pos = asset.data.joint_pos[:, joint_index].squeeze()
            current_pos = mdp.joint_pos_rel(env)[:, joint_index].squeeze()

            # calculate penalty for diverging from the expected joint trajectory
            def_pose = torch.zeros_like(current_pos)
            reward = torch.square(current_pos - joint_targets)
            rewards.append(reward)
        
        #interestingly, the reward term seems to be shape (num_envs, ), not (num_envs, 1)
        total_reward = torch.stack(rewards, dim = -1).sum(-1)
        return total_reward


#this tracks the xyz position of a joint in robot root frame
class joint_position_tracking(ManagerTermBase):

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):

        # open the trajectory data and convert to torch tensor
        traj_path = '/home/temuge/my_bots/tip_trajectory.pkl'
        with open(traj_path, "rb") as f:
            joint_traj = pickle.load(f)
        self.joint_traj = {k: torch.tensor(v, device=env.device) for k,v in joint_traj.items()}
        self.target_joints = list(joint_traj.keys())

        foot_pose = torch.tensor([0.0, -0.09, 0.013, 0.0, 0.0, 0.0, 1.0], device = env.device)
        self.foot_parent_transforms = {
            'tarsus_fl': foot_pose, 
            'tarsus_fr': foot_pose, 
            'tarsus_rl': foot_pose, 
            'tarsus_rr': foot_pose
        }

        self.foot_parent_links = {
            'tarsus_fl': 'foot_fl', 
            'tarsus_fr': 'foot_fr', 
            'tarsus_rl': 'foot_rl', 
            'tarsus_rr': 'foot_rr'
        }

    #this takes the body poses in world frame and returns them in root frame
    #additionally, it calculates the position of the feet in root frame
    def get_joint_positions_in_root_frame(self, asset: Articulation, link_pose_w: torch.Tensor):

        # get the root position and orientation
        root_pos_w = link_pose_w[:, 0:1, :3] # [num_envs, 1, 3]
        root_quat_w = link_pose_w[:, 0:1, 3:] # [num_envs, 1, 4]

        # conjugate for inverse rotation
        root_quat_conj = quat_conjugate(root_quat_w)

        # extract all link poses
        link_pos_w = link_pose_w[:, 1:, :3]       # [num_envs, num_links, 3]
        link_quat_w = link_pose_w[:, 1:, 3:]      # [num_envs, num_links, 4]

        # position in root frame
        root_quat_conj_expanded = root_quat_conj.expand(-1, link_pos_w.shape[1], 4) #needed for quat_apply
        pos_rel = quat_apply(root_quat_conj_expanded, link_pos_w - root_pos_w)

        # orientation in root frame
        quat_rel = quat_mul(root_quat_conj_expanded, link_quat_w)

        # merge
        link_pose_r = torch.cat([pos_rel, quat_rel], dim=-1)
        #print('link_pose_r:', link_pose_r.shape)

        # calculate the position of the feet
        feet_pose_r = {}
        for parent_joint, foot_pose in self.foot_parent_transforms.items():
            # Get parent index
            joint_index = asset.find_joints([parent_joint])[0]

            # Split foot offset
            p_off = foot_pose[:3]
            q_off = foot_pose[3:]

            # Parent pose in root frame
            p_parent = link_pose_r[:, joint_index, :3]
            q_parent = link_pose_r[:, joint_index, 3:]

            #explicitly match the shape for each env
            p_off = p_off.expand(p_parent.shape)
            q_off = q_off.expand(q_parent.shape)

            # Transform offset into root frame
            p_foot = p_parent + quat_apply(q_parent, p_off)
            q_foot = quat_mul(q_parent, q_off)

            # Merge into pose
            feet_pose_r[self.foot_parent_links[parent_joint]] = torch.cat([p_foot, q_foot], dim=-1)

        return link_pose_r, feet_pose_r

    def __call__(self, 
                 env: ManagerBasedRLEnv, 
                 animation_fps: float,
                 target_shift: list[float],
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset : Articulation = env.scene[asset_cfg.name]

        #Get the current time of each environment
        sampling_times = mdp.current_time_s(env).squeeze(-1)

        #get the position of each link in root frame. Also get the position of the feet in root frame
        link_pose_r, feet_pose_r = self.get_joint_positions_in_root_frame(asset, asset.data.body_link_pose_w)

        #target shift as torch tensor
        target_shift = torch.tensor(target_shift, device=env.device)

        rewards = []
        for joint_name, trajectory in self.joint_traj.items():
            num_frames = len(trajectory)

            # fractional index
            frame_idx = sampling_times * animation_fps

            # wrap animation
            frame_idx = frame_idx % num_frames

            # get the frames to the left and right
            i0 = frame_idx.floor().long()
            i1 = (i0 + 1) % num_frames

            # interpolation weight
            weights = (frame_idx - i0).unsqueeze(-1)

            # linear interpolate
            joint_targets = (1 - weights) * trajectory[i0] + weights * trajectory[i1]            
            target_shift = target_shift.expand(joint_targets.shape)
            joint_targets += target_shift

            #current position of the joint for all envs
            current_pos = feet_pose_r[joint_name][:,0,:3] # for now only the position is tracked
            
            reward = torch.square(joint_targets - current_pos).sum(-1)
            rewards.append(reward)

        total_reward = torch.stack(rewards, dim = -1).sum(-1)
        return total_reward
    
def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
