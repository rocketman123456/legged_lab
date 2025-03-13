# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.assets import ISAAC_ASSET_DIR

SROBOT_WHEEL_LEGGED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/srobot_wheel_legged/usd/srobot-wheel-legged.usd",
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
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_pitch.*": -0.28,
            ".*_knee.*": 0.79,
            ".*_ankle.*": -0.52,
            ".*_shoulder_pitch.*": 0.20,
            ".*_elbow.*": 0.32,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_hip_pitch.*", ".*_knee.*", ".*torso.*"],
            effort_limit_sim=300.0,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw.*": 200.0,
                ".*_hip_roll.*": 200.0,
                ".*_hip_pitch.*": 200.0,
                ".*_knee.*": 300.0,
                ".*torso.*": 300.0,
            },
            damping={
                ".*_hip_yaw.*": 5.0,
                ".*_hip_roll.*": 5.0,
                ".*_hip_pitch.*": 5.0,
                ".*_knee.*": 6.0,
                ".*torso.*": 6.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle.*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness={".*_ankle.*": 40.0},
            damping={".*_ankle.*": 2.0},
        ),
        # "arms": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_shoulder_pitch.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_elbow.*"],
        #     effort_limit_sim=300.0,
        #     velocity_limit_sim=100.0,
        #     stiffness={
        #         ".*_shoulder_pitch.*": 100.0,
        #         ".*_shoulder_roll.*": 50.0,
        #         ".*_shoulder_yaw.*": 50.0,
        #         ".*_elbow.*": 50.0,
        #     },
        #     damping={
        #         ".*_shoulder_pitch.*": 2.0,
        #         ".*_shoulder_roll.*": 2.0,
        #         ".*_shoulder_yaw.*": 2.0,
        #         ".*_elbow.*": 2.0,
        #     },
        # ),
    },
)
