from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseEnvCfg,
    BaseAgentCfg,
    BaseSceneCfg,
    RobotCfg,
    DomainRandCfg,
    RewardCfg,
    HeightScannerCfg,
    AddRigidBodyMassCfg,
    PhysxCfg,
    SimCfg,
    MLPPolicyCfg,
    RNNPolicyCfg,
)
from legged_lab.assets.unitree import H1_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
import legged_lab.mdp as mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class H1RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0}
    )
    fly = RewTerm(func=mdp.fly, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"), "threshold": 1.0})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped, weight=0.5, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"), "threshold": 0.4}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"), "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*")},
    )
    feet_force = RewTerm(
        func=mdp.body_force, weight=-3e-3, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"), "threshold": 500, "max_reward": 400}
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle.*"]), "threshold": 0.3}
    )
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle.*"])})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])}
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*torso.*", ".*_shoulder.*", ".*_elbow.*"])}
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*ankle.*"])}
    )


@configclass
class H1FlatEnvCfg(BaseEnvCfg):
    scene = BaseSceneCfg(
        height_scanner=HeightScannerCfg(enable_height_scan=False, prim_body_name="torso_link"),
        robot=H1_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
    )
    robot = RobotCfg(terminate_contacts_body_names=[".*torso.*"], feet_body_names=[".*ankle.*"])
    domain_rand = DomainRandCfg(
        add_rigid_body_mass=AddRigidBodyMassCfg(enable=True, params={"body_names": [".*torso.*"], "mass_distribution_params": (-5.0, 5.0), "operation": "add"})
    )
    reward = H1RewardCfg()


@configclass
class H1FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "h1_flat"
    wandb_project: str = "h1_flat"


@configclass
class H1RoughEnvCfg(H1FlatEnvCfg):
    scene = BaseSceneCfg(
        height_scanner=HeightScannerCfg(enable_height_scan=True, prim_body_name="torso_link"),
        robot=H1_CFG,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
    )
    robot = RobotCfg(actor_obs_history_length=1, critic_obs_history_length=1, terminate_contacts_body_names=[".*torso.*"], feet_body_names=[".*ankle.*"])
    reward = H1RewardCfg(
        track_lin_vel_xy_exp=RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.5}),
        track_ang_vel_z_exp=RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5}),
        lin_vel_z_l2=RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25),
    )


@configclass
class H1RoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "h1_rough"
    wandb_project: str = "h1_rough"
    policy = RNNPolicyCfg()
