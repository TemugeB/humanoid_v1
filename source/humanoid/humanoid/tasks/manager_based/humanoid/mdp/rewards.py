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
from isaaclab.sensors import ContactSensor

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
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)
    

class joint_angle_tracking(ManagerTermBase):

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):

        # open the trajectory data and convert to torch tensor
        #traj_path = '/home/temuge/isaac_projects/humanoid_v1/animation/robot_angles.pkl'
        traj_path = '/home/temuge/isaac_projects/humanoid_v1/animation/postprocessed_joint_angles.pkl'
        with open(traj_path, "rb") as f:
            joint_traj = pickle.load(f)
        self.joint_traj = {k: torch.tensor(v, device=env.device) for k,v in joint_traj.items()}

        self.tracking_weight = {
            'hip_left_x'        : 1.0, 
            'hip_left_z'        : 1.0,
            'hip_left_y'        : 1.0,
            'knee_left'         : 0.1,
            'left_ankle'        : 1.0,
            'hip_right_x'       : 1.0, 
            'hip_right_z'       : 1.0,
            'hip_right_y'       : 1.0,
            'knee_right'        : 0.1,
            'right_ankle'       : 1.0,
            'spine'             : 1.0,
            'left_shoulder_y'   : 1.0, 
            'left_shoulder_x'   : 1.0, 
            'left_shoulder_z'   : 1.0,
            'right_shoulder_y'  : 1.0, 
            'right_shoulder_x'  : 1.0, 
            'right_shoulder_z'  : 1.0,
            'left_elbow'        : 1.0,
            'right_elbow'       : 1.0
        }

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
            rewards.append(self.tracking_weight[joint_name] * reward)
        
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

class default_pose_tracking(ManagerTermBase):

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

        default_pose = {
            'hip_left_x'        : 0.0, 
            'hip_left_z'        : 0.0,
            'hip_left_y'        : 0.0,
            'knee_left'         : 0.0,
            'left_ankle'        : 0.0,
            'hip_right_x'       : 0.0, 
            'hip_right_z'       : 0.0,
            'hip_right_y'       : 0.0,
            'knee_right'        : 0.0,
            'right_ankle'       : 0.0,
            'spine'             : 0.0,
            'left_shoulder_y'   : 0.0, 
            'left_shoulder_x'   : 0.0, 
            'left_shoulder_z'   : 0.0,
            'right_shoulder_y'  : 0.0, 
            'right_shoulder_x'  : 0.0, 
            'right_shoulder_z'  : 0.0,    
        }

        asset : Articulation = env.scene[asset_cfg.name]
        
        #get the joint indices
        joint_inds = asset.find_joints(default_pose.keys())[0]
        self.joint_inds = torch.tensor(joint_inds).reshape(-1)
        #get the target joint values for each joint
        self.joint_targets = torch.tensor(list(default_pose.values()), device=env.device, dtype=torch.float32).reshape(1, -1)


    def __call__(self, 
                 env: ManagerBasedRLEnv, 
                 threshold: float,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        
        asset : Articulation = env.scene[asset_cfg.name]
        joint_pos = asset.data.joint_pos[:, self.joint_inds]
        #which ones are over threshold
        joint_errors = joint_pos - self.joint_targets # [num_envs, num_joints]
        over_threshold = torch.abs(joint_errors) > threshold

        return torch.sum(torch.square(joint_errors * over_threshold), dim=1)

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def root_motion_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    root_velocities = mdp.base_lin_vel(env, asset_cfg)
    return torch.sum(torch.square(root_velocities), dim=1)

def root_rotation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    root_angular_velocities = mdp.base_ang_vel(env, asset_cfg)
    return torch.sum(torch.square(root_angular_velocities), dim=1)


class action_acc_l2(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):

        action_shape = env.action_manager.action.shape # Get shape from the first action
        device = env.device # Get device from the environment

        self.last_action = torch.zeros(action_shape, device=device)
        self.last_last_action = torch.zeros(action_shape, device=device)

    def __call__(self, 
                 env: ManagerBasedRLEnv) -> torch.Tensor:

        current_action = env.action_manager.action
        action_diff = current_action - 2 * self.last_action + self.last_last_action
        reward_term = torch.square(torch.sum(action_diff, dim=-1))

        print(self.last_last_action[0][0], self.last_action[0][0], current_action[0][0])
        self.last_last_action = self.last_action
        self.last_action = current_action

        return reward_term
    

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_motion(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1), dim=1)
    return reward

def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


class compound_reward(ManagerTermBase):

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

        default_pose = {
            'hip_left_x'        : 0.0, 
            'hip_left_z'        : 0.0,
            'hip_left_y'        : 0.0,
            'knee_left'         : 0.0,
            'left_ankle'        : 0.0,
            'hip_right_x'       : 0.0, 
            'hip_right_z'       : 0.0,
            'hip_right_y'       : 0.0,
            'knee_right'        : 0.0,
            'right_ankle'       : 0.0,
            'spine'             : 0.0,
            'left_shoulder_y'   : 0.0, 
            'left_shoulder_x'   : 0.0, 
            'left_shoulder_z'   : 0.0,
            'right_shoulder_y'  : 0.0, 
            'right_shoulder_x'  : 0.0, 
            'right_shoulder_z'  : 0.0, 
            'left_elbow'        : 0.0,
            'right_elbow'       : 0.0,   
        }

        self.asset : Articulation = env.scene[asset_cfg.name]
        
        #get the joint indices
        joint_inds = self.asset.find_joints(default_pose.keys())[0]
        self.joint_inds = torch.tensor(joint_inds).reshape(-1)
        #get the target joint values for each joint
        self.joint_targets = torch.tensor(list(default_pose.values()), device=env.device, dtype=torch.float32).reshape(1, -1)

        #default reward weights.
        self.weights = {
            'alive': 10.0,
            'torque_usage': -1e-3,
            'joint_accel': -2.5e-6,
            'action_rate': -1.5,
            'pose_tracking': -2.0
        }
        
        #container to hold the decayed weights. Initializing here is just an optimization.
        self.term_decay = {
            'alive':        torch.ones((env.num_envs,), device=env.device),
            'torque_usage': torch.ones((env.num_envs,), device=env.device),
            'joint_accel':  torch.ones((env.num_envs,), device=env.device),
            'action_rate':  torch.ones((env.num_envs,), device=env.device),
            'pose_tracking':torch.ones((env.num_envs,), device=env.device),
        }

        self.current_weights = {
            'alive': torch.tensor([1.0, 1.0]),        #[current_value, final_value]
            'torque_usage': torch.tensor([1.0, 1.0]), 
            'joint_accel': torch.tensor([1.0, 1.0]),
            'action_rate': torch.tensor([1.0, 1.0]),
            'pose_tracking': torch.tensor([1.0, 5.0])
        }

    def __call__(self, 
                 env: ManagerBasedRLEnv, 
                 threshold: float) -> torch.Tensor:
        
        joint_pos = self.asset.data.joint_pos[:, self.joint_inds]
        joint_errors = joint_pos - self.joint_targets # [num_envs, num_joints]
        pose_tracking = torch.mean(torch.square(joint_errors), dim=1)
        
        #modify the reward weight for this iteration
        self._update_weight_decay(env, pose_tracking, threshold)

        #reward terms
        alive =         self.weights['alive']         * self.term_decay['alive'] * mdp.is_alive(env)
        torque_usage =  self.weights['torque_usage']  * self.term_decay['torque_usage'] * mdp.joint_torques_l2(env)
        joint_accel  =  self.weights['joint_accel']   * self.term_decay['joint_accel'] * mdp.joint_acc_l2(env)
        action_rate =   self.weights['action_rate']   * self.term_decay['action_rate'] * mdp.action_rate_l2(env)
        pose_tracking=  self.weights['pose_tracking'] * self.term_decay['pose_tracking'] * pose_tracking

        return alive + torque_usage + joint_accel + action_rate + pose_tracking
    
    def _update_weight_decay(self, env, pose_tracking, threshold):
        
        #common_step_counter is the number of policy updates.
        total_steps = env.common_step_counter * env.num_envs
        
        # Define curriculum parameters based on total steps
        P_start = 250_000_000  # Start curriculum after 250M total steps
        P_duration = 750_000_000 # Decay smoothly over the next 500M total steps

        # alpha smoothly increases from 0.0 to 1.0 during the curriculum
        current_steps_in_curriculum = torch.clamp(
            torch.tensor(float(total_steps - P_start), device=env.device),
            0.0,
            float(P_duration)
        )
        alpha = current_steps_in_curriculum / P_duration

        for key in self.current_weights.keys():
            # Get start weight (1.0) and final weight (self.current_weights[key][1])
            start_w = 1.0
            # Ensure final_w is a standard Python float for calculation
            final_w = self.current_weights[key][1].item() 

            # Linear Interpolation: (1 - alpha) * start + alpha * final
            # Calculate the new target weight
            new_weight = (1.0 - alpha) * start_w + alpha * final_w
            
            # Update the decay tensor for ALL environments using fill_ (highly efficient)
            self.term_decay[key].fill_(new_weight)
    
        policy_steps = total_steps // env.num_envs
        if policy_steps % (500_000_000 // env.num_envs // 50) == 0:
             print(f"Total Steps: {total_steps}. Curriculum Alpha: {alpha.item():.4f}. Alive Weight: {self.term_decay['alive'][0].item():.4f}")

        return


class feet_contact_tracking(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):

        num_frames = 135
        #get the indices masks for contact penalty
        left_no_contact_frames = list(range(54, 86))
        self.left_nocontact_mask = torch.zeros((num_frames), dtype=torch.int32, device = env.device)
        self.left_nocontact_mask[left_no_contact_frames] = 1

        right_no_contact_frames = list(range(18)) + list(range(121, 135))
        self.right_nocontact_mask = torch.zeros((num_frames), dtype=torch.int32, device = env.device)
        self.right_nocontact_mask[right_no_contact_frames] = 1

        contact_sensor = env.scene.sensors['contact_forces']
        self.left_feet_id = contact_sensor.find_bodies(['shoes'])[0]
        self.right_feet_id = contact_sensor.find_bodies(['shoes_2'])[0]

    def __call__(self, 
                 env: ManagerBasedRLEnv,
                 animation_fps: float,
                 num_frames: int,
                 ) -> torch.Tensor:
    
        contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        
        #get the contact state of the left feet. Has shape [num_envs, 1]
        left_contacts = contact_sensor.data.net_forces_w_history[:, :, self.left_feet_id, :].norm(dim=-1).max(dim=1)[0] > 1.0
        right_contacts = contact_sensor.data.net_forces_w_history[:, :, self.right_feet_id, :].norm(dim=-1).max(dim=1)[0] > 1.0

        #get the current frame index of each environment
        sampling_times = mdp.current_time_s(env).squeeze(-1)

        # fractional index
        frame_idx = sampling_times * animation_fps

        # wrap animation
        frame_idx = frame_idx % num_frames

        # find the closes frame index
        frame_idx = torch.floor(frame_idx).to(torch.int32)

        # if current frame prohibits contact. [num_envs, 1]
        left_contact_prohibited = self.left_nocontact_mask[frame_idx].reshape(-1, 1)
        right_contact_prohibited = self.right_nocontact_mask[frame_idx].reshape(-1, 1)

        # currently in illegal contact?
        left_illegal_foot_contact = left_contacts * left_contact_prohibited
        right_illegal_foot_contact = right_contacts * right_contact_prohibited

        # penalty term
        foot_contact_tracking_penalty = left_illegal_foot_contact + right_illegal_foot_contact
        return foot_contact_tracking_penalty.squeeze()

class joint_velocity_tracking(ManagerTermBase):

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):

        # open the trajectory data and convert to torch tensor
        traj_path = '/home/temuge/isaac_projects/humanoid_v1/animation/postprocessed_joint_velocities.pkl'
        with open(traj_path, "rb") as f:
            joint_traj = pickle.load(f)
        self.joint_velocities = {k: torch.tensor(v, device=env.device) for k,v in joint_traj.items()}

        self.tracking_weight = {
            'hip_left_x'        : 1.0, 
            'hip_left_z'        : 1.0,
            'hip_left_y'        : 1.0,
            'knee_left'         : 0.1,
            'left_ankle'        : 1.0,
            'hip_right_x'       : 1.0, 
            'hip_right_z'       : 1.0,
            'hip_right_y'       : 1.0,
            'knee_right'        : 0.1,
            'right_ankle'       : 1.0,
            'spine'             : 1.0,
            'left_shoulder_y'   : 1.0, 
            'left_shoulder_x'   : 1.0, 
            'left_shoulder_z'   : 1.0,
            'right_shoulder_y'  : 1.0, 
            'right_shoulder_x'  : 1.0, 
            'right_shoulder_z'  : 1.0,
            'left_elbow'        : 1.0,
            'right_elbow'       : 1.0
        }

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
        for joint_name, curve in self.joint_velocities.items():
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
            joint_v_targets = (1 - weights) * curve[i0] + weights * curve[i1]
            # to seconds
            joint_v_targets *= animation_fps

            # get the current joint positions
            joint_index = asset.find_joints([joint_name])[0]
            #current_pos = asset.data.joint_pos[:, joint_index].squeeze()
            current_rot_vel = mdp.joint_vel_rel(env)[:, joint_index].squeeze()

            # calculate penalty for diverging from the expected joint velocity
            reward = torch.square(current_rot_vel - joint_v_targets)
            rewards.append(self.tracking_weight[joint_name] * reward)
        
        #interestingly, the reward term seems to be shape (num_envs, ), not (num_envs, 1)
        total_reward = torch.stack(rewards, dim = -1).sum(-1)
        return total_reward

def lateral_motion(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    
    asset : Articulation = env.scene[asset_cfg.name]
    vel_base = asset.data.root_com_lin_vel_b
    lateral_vel = vel_base[:, 1].reshape(-1)
    return torch.abs(lateral_vel)