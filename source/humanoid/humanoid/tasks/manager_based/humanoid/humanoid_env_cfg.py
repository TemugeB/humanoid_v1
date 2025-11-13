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


joints = [
    'hip_left_y', 'hip_left_x', 'hip_left_z', 'knee_left', 'left_ankle',
    'hip_right_y', 'hip_right_x', 'hip_right_z', 'knee_right', 'right_ankle',
    'spine',
    'left_shoulder_y', 'left_shoulder_x', 'left_shoulder_z', 'left_elbow',
    'right_shoulder_y', 'right_shoulder_x', 'right_shoulder_z', 'right_elbow'    
]

usd_path = "/home/temuge/my_bots/humanoid_urdf/robot/humanoid.usd"
usd_path = "/home/temuge/isaac_projects/my_bots/humanoid_urdf/robot/humanoid.usd"

HUMANOID_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        # joint_pos={
        #     "hip_left_y":  0.31,
        #     "hip_right_y": -0.31,
        #     "knee_left": -0.34,
        #     "knee_right": 0.34, 
        #     "left_ankle": 0.32,
        #     "right_ankle": -0.32,
        # },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[    
                'hip_left_y', 'hip_left_x', 'hip_left_z', 'knee_left',
                'hip_right_y', 'hip_right_x', 'hip_right_z', 'knee_right'],
            effort_limit_sim=300,
            stiffness={
                "hip_left_y":  150.0,
                "hip_left_x":  150.0,
                "hip_left_z":  150.0,
                "hip_right_y": 150.0,
                "hip_right_x": 150.0,
                "hip_right_z": 150.0,
                "knee_left":   150.0,
                "knee_right":  150.0
            },
            damping={
                "hip_left_y":  10.0,
                "hip_left_x":  10.0,
                "hip_left_z":  10.0,
                "hip_right_y": 10.0,
                "hip_right_x": 10.0,
                "hip_right_z": 10.0,
                "knee_left":   10.0,
                "knee_right":  10.0
            },
            armature = 0.065

        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["left_ankle", "right_ankle"],
            effort_limit_sim=300,
            stiffness={"left_ankle":  100.0,
                       "right_ankle": 100.0},
            damping={"left_ankle":    10.0,
                     "right_ankle":   10.0},
            armature = 0.065
        ),
        "spine": ImplicitActuatorCfg(
            joint_names_expr=["spine"],
            effort_limit_sim=300.,
            stiffness={"spine": 130.0},
            damping={"spine": 10.0},
            armature = 0.065

        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=['left_shoulder_y', 'left_shoulder_x', 'left_shoulder_z', 'left_elbow',
                              'right_shoulder_y', 'right_shoulder_x', 'right_shoulder_z', 'right_elbow'],
            effort_limit_sim=300.,
            stiffness={
                "left_shoulder_y":  100.0,
                "left_shoulder_x":  100.0,
                "left_shoulder_z":  100.0,
                "left_elbow":       100.0,
                "right_shoulder_y": 100.0,
                "right_shoulder_x": 100.0,
                "right_shoulder_z": 100.0,
                "right_elbow":      100.0,
            },
            damping={
                "left_shoulder_y":  10.0,
                "left_shoulder_x":  10.0,
                "left_shoulder_z":  10.0,
                "left_elbow":       10.0,
                "right_shoulder_y": 10.0,
                "right_shoulder_x": 10.0,
                "right_shoulder_z": 10.0,
                "right_elbow":      10.0,
            },
            armature = 0.065

        ),
    },
)


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
            static_friction=10.0,
            dynamic_friction=10.0,
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


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=joints, scale = 1.0, use_default_offset = True)


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
        base_orientation = ObsTerm(func=mdp.base_yaw_pitch_roll)
        #base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        #base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        #joint_acc_rel = ObsTerm(func=mdp.joint_acceleration, scale = 0.1)
        actions = ObsTerm(func=mdp.last_action)
        feet_contacts = ObsTerm(func=mdp.feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["shoes", "shoes_2"])})
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
        params={
            "pose_range": {},
            "velocity_range": {}
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # reset_joint_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode='reset',
    #     params={
    #         'force_range': (-.3, .3),
    #         'torque_range': (-.3, .3)
    #     }
    # )

    # reset_root_velocity = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode='reset',
    #     params={
    #         'velocity_range': {'x': (-0.65, 0.65),
    #                            'y': (-0.65, 0.65)}
    #     }
    # )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.65, 0.65),
    #             "y": (-0.65, 0.65),
    #             "z": (-0.65, 0.65),
    #             "roll": (-0.5, 0.5),
    #             "pitch": (-0.5, 0.5),
    #             "yaw": (-0.5, 0.5),
    #         },
    #     },
    # )


    pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    #termination_penalty = RewTerm(func=mdp.is_terminated,weight=0.0,)
    #upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    #normal_pose = RewTerm(func=mdp.joint_pos_target_l2, weight=-0.5, params={'target': 0.0})
    default_pose = RewTerm(func=mdp.default_pose_tracking, weight = -0.5, params={'threshold': 0.0})
    #stadning_pose = RewTerm(func=mdp.standing_pose_tracking, weight = -0.5)
    torque_usage = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3)
    joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)
    #action_l2 = RewTerm(func=mdp.action_l2, weight = -0.5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.5)
    #action_accel_l2 = RewTerm(func=mdp.action_acc_l2, weight=-0.45)
    #root_motion_l2 = RewTerm(func=mdp.root_motion_l2, weight = -3.0)
    #root_rotation_l2 = RewTerm(func=mdp.root_rotation_l2, weight = -0.001)
    feet_slide = RewTerm(func=mdp.feet_slide, weight = -1., params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["shoes", "shoes_2"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["shoes", "shoes_2"]) })
    #ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # terminate if body part touches the ground
    unwanted_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", 
                                             body_names=['hip', 'connector0_3', 'connector0_4', 'connector0_5', 
                                                         'connector0_2', 'connector0', 'connector3', 'connector3_2',
                                                         'connector7', 'connector8', 'connector7_2', 'connector8_2']), "threshold": 1.0},
    )


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
        self.decimation = 4
        self.episode_length_s = 8
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1/200.
        self.sim.render_interval = self.decimation