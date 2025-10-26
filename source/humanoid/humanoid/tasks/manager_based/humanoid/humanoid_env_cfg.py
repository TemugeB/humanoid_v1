# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg


from . import mdp

##
# Pre-defined configs
##

#Tested following fps: 15, 24, 36, 48
animation_fps = 24.0
num_frames = 30
damping_val = 0.5
stiffness_val = 25.0
joints = [
    'hip_left_y', 'hip_left_x', 'hip_left_z', 'knee_left', 'left_ankle',
    'hip_right_y', 'hip_right_x', 'hip_right_z', 'knee_right', 'right_ankle',
    'spine',
    'left_shoulder_y', 'left_shoulder_x', 'left_shoulder_z', 'left_elbow',
    'right_shoulder_y', 'right_shoulder_x', 'right_shoulder_z', 'right_elbow'    
]

#usd_path = "/home/temuge/robots/spiderbot/robot_w_tip/spiderbot.usd"
usd_path = "/home/temuge/my_bots/humanoid/robot/humanoid.usd"
HUMANOID_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=usd_path,
                               activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.62)),
    actuators={"actuators": ImplicitActuatorCfg(joint_names_expr=joints, 
                                                effort_limit = 80.0,
                                                velocity_limit = 15.0,
                                                damping=damping_val, 
                                                stiffness=stiffness_val, 
                                                )},
)

##
# Scene definition
##


@configclass
class HumanoidSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.5,
        ),
        visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = HUMANOID_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/humanoid/.*", history_length=3, track_air_time=True)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=joints, scale=1.)

    #joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=joints, scale = 0.25, use_default_offset = True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        #base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        actions = ObsTerm(func=mdp.last_action)
        #motion_phase = ObsTerm(func=mdp.motion_phase_observation, params={'animation_fps': animation_fps, 'num_frames': num_frames})



        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.2, 0.2),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # terminate if body part touches the ground
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", 
                                             body_names=['hip', 'connector0_3', 'connector0_4', 'connector0_5', 
                                                         'connector0_2', 'connector0', 'connector3', 'connector3_2',
                                                         'connector7', 'connector8', 'connector7_2', 'connector8_2']), "threshold": 1.0},
    )



##
# Environment configuration
##


@configclass
class HumanoidEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: HumanoidSceneCfg = HumanoidSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation