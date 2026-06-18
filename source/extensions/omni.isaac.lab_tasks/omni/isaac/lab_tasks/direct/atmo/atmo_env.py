# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from numpy import pi, exp, copy
from IPython import embed

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_rotate, matrix_from_quat
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor

##
# Pre-defined configs
##
from omni.isaac.lab_assets import ATMO_CFG            # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class ATMOEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: ATMOEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class ATMOEnvCfg(DirectRLEnvCfg):
    # high level flags
    randomize = True
    terminate = True
    disturb = True
    noise = True
    actuator_dynamics = True
    randomize_motor_dynamics = True
    quantize_tilt_action = True

    # curriculum stage selection: 1 = vertical-thrust landing, 2 = morpholanding
    curriculum_stage = 2
    rotate_rotor_thrust_by_stage = [False, True]
    belly_contact_is_death_by_stage = [False, True]
    contact_reward_requires_xy_by_stage = [True, False]

    curriculum_update_rate = 8e3
    curriculum_steps_to_completion = curriculum_update_rate * 10

    # action history
    action_history_length = 10
    observation_history_length = 24

    # env
    episode_length_s = 5.0
    sim_dt = 1 / 50   # training 1/100
    decimation = 1      # training 2
    action_space = 5

    num_obs = 19
    observation_space = ((observation_history_length + 1) * num_obs) + (action_space * action_history_length)

    num_privileged_obs = 12
    state_space = 0
    debug_vis = True
    num_envs = 32768

    ui_window_class_type = ATMOEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ATMO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", track_air_time=True, history_length=2
    )

    # termination conditions
    too_fast_vel = [2.0, 2.0]
    termination_dxy = [1.50, 3.00]
    termination_height = [3.0, 3.0]

    # observation delay (num steps)
    observation_buffer_length = 25
    observation_delay = 0

    # acceptance state radius
    delta_d = [0.40, 0.60]

    # desired velocity
    vx_des, vy_des, vz_des = 0.0, 0.0, -0.50

    # reward scales
    lin_vel_pen_scale = [-0.10, -0.10]
    ang_vel_pen_scale = [-0.30, -0.30]
    spin_pen_scale = [-0.30, -0.30]
    action_rate_pen_scale = [-0.80, -0.30]
    ground_thrust_pen_scale = [-0.13, -0.13]
    orientation_pen_scale = [-0.10, -0.10]

    impulse_pen = [-1.0, -0.25]
    died_pen = [-10.0, -10.0]
    timeout_pen = [-2.0, -32.0]

    distance_to_goal_xy_rew_scale = [0.30, 0.30]
    descending_rew_scale = [0.30, 0.60]
    tilt_rew_scale = [0.80, 0.80]
    contact_in_acceptance_rew_scale = [0.40, 4.00]
    touchdown_rew_scale = [0.00, 3.00]

    # nominal parameters
    kT_0 = 28.15
    kM_0 = 0.018
    max_tilt_vel_0 = pi / 8

    # random force and torque scales
    disturbance_force_scale = 4 * kT_0 * 0.50        # best 0.05
    disturbance_moment_scale = 4 * kT_0 * kM_0 * 0.05  # best 0.05

    dist_force_cts_scale = 4 * kT_0 * 0.0
    dist_moment_cts_scale = 4 * kT_0 * kM_0 * 0.0

    # randomization parameters
    kT_error_scale = 0.2
    kM_error_scale = 0.2
    max_tilt_vel_error_scale = 0.2
    initial_height_range = [1.0, 2.0]
    initial_lin_vel_range = [-0.1, 0.1]
    initial_ang_vel_range = [-0.1, 0.1]
    initial_tilt_range = [0.0, pi / 6]
    initial_tilt_vel_range = [0.0, 1.0]

    # low pass filter constant
    step_dt = sim_dt * decimation
    T_m_range = [0.1, 0.2]
    T_m_0 = 0.15
    alpha_0 = 1.0 - exp(-step_dt / T_m_0).item()
    alpha_range = [1.0 - exp(-step_dt / T_m_range[1]).item(), 1.0 - exp(-step_dt / T_m_range[0]).item()]

    # observation noise scales
    pos_noise_scale = 0.005                  # 0.5 cm
    quat_noise_scale = 0.005                  # 0.5 percent
    lin_vel_noise_scale = 0.035                  # 0.035 m/s
    ang_vel_noise_scale = 0.035                  # 2 deg/s or 0.035 rad/s
    tilt_noise_scale = 0.018                  # 2 degrees or 0.18 rad/s
    roll_noise_scale = 0.008                  # 2 degrees or 0.18 rad/s
    pitch_noise_scale = 0.008                  # 2 degrees or 0.18 rad/s
    rot_noise_scale = 0.005                  # 0.5 percent


class ATMOEnv(DirectRLEnv):
    cfg: ATMOEnvCfg

    def __init__(self, cfg: ATMOEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._observation_buffer = torch.zeros(self.num_envs, self.cfg.observation_buffer_length, self.cfg.num_obs, device=self.device)

        self.box_extent = 0.1
        self.curriculum_update_time = 0
        self.distance_to_goal_epoch_av = 0.0

        self._current_contacts = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._in_acceptance_ball = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._in_acceptance_xy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # desired velocity
        self._lin_vel_des = torch.zeros(3, device=self.device)
        self._lin_vel_des[0], self._lin_vel_des[1], self._lin_vel_des[2] = self.cfg.vx_des, self.cfg.vy_des, self.cfg.vz_des

        # Initialize the action space
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._actions_filtered = torch.zeros_like(self._actions)
        self._previous_actions = torch.zeros_like(self._actions)

        # Record an action history
        self._action_history = torch.zeros(self.num_envs, self.cfg.action_history_length, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Thrust and moments applied to the rotor bodies
        self.thrust = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.moment = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # Joint velocities and positions
        self._tilt_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self._tilt_angle = torch.zeros(self.num_envs, 1, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Total impulse
        self._current_impulse = torch.zeros(self.num_envs, 1, device=self.device)

        # Acceleration
        self._previous_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._acceleration = torch.zeros(self.num_envs, 3, device=self.device)

        # First contact flag (all environments start with False)
        self._first_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # disturbance force and moment
        self._disturbance_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._disturbance_moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # push time
        self._push_time = torch.zeros(self.num_envs, device=self.device)
        self._push_duration = torch.zeros(self.num_envs, device=self.device)

        # time in each environment
        self._time_elapsed = torch.zeros(self.num_envs, device=self.device)

        # filter alpha
        self._alpha = self.cfg.alpha_0 * torch.ones(self.num_envs, 1, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel_pen",
                "ang_vel_pen",
                "action_rate_pen",
                "ground_thrust_penalty",
                "spin_penalty",
                "orientation_pen",
                "died_penalty",
                "impulse_penalty",
                "distance_to_goal_xy_rew",
                "descending_rew",
                "tilt_rew",
                "touchdown_rew",
                "contact_in_acceptance_rew",
                "timeout_high_penalty",
            ]
        }

        # Get specific body indices
        self._base_link = self._robot.find_bodies("base_link")[0][0]
        self._rotor0 = self._robot.find_bodies("rotor0")[0][0]
        self._rotor1 = self._robot.find_bodies("rotor1")[0][0]
        self._rotor2 = self._robot.find_bodies("rotor2")[0][0]
        self._rotor3 = self._robot.find_bodies("rotor3")[0][0]

        # Get the joint indices
        self._joint0 = self._robot.find_joints("base_to_arml")[0][0]
        self._joint1 = self._robot.find_joints("base_to_armr")[0][0]

        # Get arml and armr indices
        self._arml = self._robot.find_bodies("arml")[0][0]
        self._armr = self._robot.find_bodies("armr")[0][0]

        # Get inertial parameters
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Initialize kT, kM and max_tilt_vel_0
        self.kT = self.cfg.kT_0 * torch.ones(self.num_envs, 1, device=self.device)
        self.kM = self.cfg.kM_0 * torch.ones(self.num_envs, 1, device=self.device)
        self.max_tilt_vel = self.cfg.max_tilt_vel_0 * torch.ones(self.num_envs, 1, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self._wheel_contact_ids = [
            self._contact_sensor.body_names.index("wheel0"),
            self._contact_sensor.body_names.index("wheel1"),
            self._contact_sensor.body_names.index("wheel2"),
            self._contact_sensor.body_names.index("wheel3"),
        ]

        self._belly_contact_ids = [
            self._contact_sensor.body_names.index("base_link"),
        ]

        self._bad_contact_ids = [
            self._contact_sensor.body_names.index("arml"),
            self._contact_sensor.body_names.index("armr"),
        ]

        self._ep_contact_in_acceptance = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._is_first_sim_step = True

    def _stage_value(self, values, name: str):
        try:
            stage = int(self.cfg.curriculum_stage)
            if stage < 1:
                raise ValueError(f"curriculum_stage must be >= 1, got {self.cfg.curriculum_stage}")
            stage_idx = stage - 1
            return values[stage_idx]
        except TypeError:
            return values
        except IndexError as exc:
            raise ValueError(
                f"curriculum_stage={self.cfg.curriculum_stage} has no entry for {name}"
            ) from exc

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):

        # increment elapsed time
        self._time_elapsed += self.step_dt

        # Clamp actions to [0, 1] even though sigmoid already achieves this
        self._actions = actions.clone().clamp(0.0, 1.0)

        # Pass thruster actions through a low pass filter
        if self.cfg.actuator_dynamics:
            self._actions_filtered[:, :4] = self._alpha * self._actions[:, :4] + (1 - self._alpha) * self._actions_filtered[:, :4]
            self._actions_filtered[:, 4] = self._actions[:, 4]
        else:
            self._actions_filtered = self._actions

        if self.cfg.quantize_tilt_action:
            self._actions_filtered[:, 4] = torch.round(self._actions_filtered[:, 4])

        # Assign the thrust to each of the rotors
        spin_direction = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=self.device)
        self.thrust[:, :, 2] = (self.kT.reshape(self.num_envs, 1, 1) * self._actions_filtered.reshape(self.num_envs, 1, 5)[:, :, :4]).squeeze()
        self.moment[:, :, 2] = spin_direction * self.kM * self.thrust[:, :, 2]

        # Assign the joint positions and velocities
        tilt_action = self._actions_filtered[:, 4].unsqueeze(1)
        self._tilt_angle = self._tilt_angle + self.max_tilt_vel * tilt_action * self.physics_dt
        self._tilt_angle = torch.clamp(self._tilt_angle, 0.0, torch.pi / 2)
        self._tilt_vel = self.max_tilt_vel * tilt_action

    def _apply_action(self):

        dist_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        dist_moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        dist_force_cts = torch.zeros(self.num_envs, 1, 3, device=self.device)
        dist_moment_cts = torch.zeros(self.num_envs, 1, 3, device=self.device)
        if self.cfg.disturb:
            # Determine whether to push robot
            push = torch.logical_and(self._time_elapsed >= self._push_time, self._time_elapsed <= self._push_time + self._push_duration).reshape(self.num_envs, 1, 1)

            # Compute disturbance force
            dist_force = self._disturbance_force * push
            dist_moment = self._disturbance_moment * push

            # Apply another disturbance force and moment of different nature
            dist_force_cts = torch.zeros(self.num_envs, 1, 3, device=self.device).uniform_(-self.cfg.dist_force_cts_scale, self.cfg.dist_force_cts_scale)
            dist_moment_cts = torch.zeros(self.num_envs, 1, 3, device=self.device).uniform_(-self.cfg.dist_moment_cts_scale, self.cfg.dist_moment_cts_scale)

        total_environmental_force = dist_force + dist_force_cts
        total_environmental_moment = dist_moment + dist_moment_cts

        if self._stage_value(self.cfg.rotate_rotor_thrust_by_stage, "rotate_rotor_thrust_by_stage"):
            rotor_ids = [self._rotor0, self._rotor1, self._rotor2, self._rotor3]
            rotor_quats = self._robot.data.body_quat_w[:, rotor_ids, :]

            q_flat = rotor_quats.reshape(-1, 4)
            thrust_flat = self.thrust.reshape(-1, 3)
            moment_flat = self.moment.reshape(-1, 3)

            thrust_world_flat = quat_rotate(q_flat, thrust_flat)
            moment_world_flat = quat_rotate(q_flat, moment_flat)

            thrust_world = thrust_world_flat.reshape(self.num_envs, 4, 3)
            moment_world = moment_world_flat.reshape(self.num_envs, 4, 3)
        else:
            thrust_world = self.thrust
            moment_world = self.moment

        # Get total force and moment across all 5 bodies (using the new rotated tensors)
        total_force = torch.cat([thrust_world, total_environmental_force], dim=1)
        total_moment = torch.cat([moment_world, total_environmental_moment], dim=1)

        # Apply forces, moments and joint targets to PhysX
        self._robot.set_external_force_and_torque(total_force, total_moment, body_ids=[self._rotor0, self._rotor1, self._rotor2, self._rotor3, self._base_link])
        self._robot.set_joint_velocity_target(self._tilt_vel.repeat(1, 2), joint_ids=[self._joint0, self._joint1])
        self._robot.set_joint_position_target(self._tilt_angle.repeat(1, 2), joint_ids=[self._joint0, self._joint1])

    def _get_observations(self) -> dict:
        self._action_history = torch.cat([self._actions.clone().unsqueeze(dim=1), self._action_history[:, :-1]], dim=1)
        relative_pos_w = self._desired_pos_w - self._robot.data.root_link_pos_w
        tilt_angle = self._robot.data.joint_pos[:, self._joint0].unsqueeze(dim=1)
        impulse = self.step_dt * torch.sum(torch.linalg.norm((self._contact_sensor.data.net_forces_w_history[:, 1, :, :] - self._contact_sensor.data.net_forces_w_history[:, 0, :, :]) * self.step_dt, dim=-1), dim=1).unsqueeze(dim=1)  # type: ignore
        self._current_impulse = impulse
        rot = matrix_from_quat(self._robot.data.root_link_quat_w)
        rot_vector = rot.reshape(-1, 9)

        # 1. Kinematic-only noise vector (19 elements)
        noise_kinematic = torch.cat(
            [
                self.cfg.pos_noise_scale * torch.zeros_like(relative_pos_w).uniform_(-1, 1),
                self.cfg.rot_noise_scale * torch.zeros_like(rot_vector).uniform_(-1, 1),
                self.cfg.lin_vel_noise_scale * torch.zeros_like(self._robot.data.root_com_lin_vel_w).uniform_(-1, 1),
                self.cfg.ang_vel_noise_scale * torch.zeros_like(self._robot.data.root_com_ang_vel_b).uniform_(-1, 1),
                self.cfg.tilt_noise_scale * torch.zeros_like(tilt_angle).uniform_(-1, 1),
            ],
            dim=-1,
        )

        # 2. Kinematic-only features (19 elements)
        obs_kinematic = torch.cat(
            [
                relative_pos_w,
                rot_vector,
                self._robot.data.root_com_lin_vel_w,
                self._robot.data.root_com_ang_vel_b,
                tilt_angle,
            ],
            dim=-1,
        )
        obs_kinematic_current = obs_kinematic + self.cfg.noise * noise_kinematic

        # 3. GPU-Native Sliding Window Step (Applied strictly to the 19 kinematic items)
        self._observation_buffer[:, 1:] = self._observation_buffer[:, :-1].clone()
        self._observation_buffer[:, 0] = obs_kinematic_current

        # 🛠️ FIXED: Changed to == 1. This clears old tracking coordinates on the fresh rollout 
        # step, preventing trajectory history pollution from breaking PPO weight updates.
        reset_env_ids = (self.episode_length_buf == 1).nonzero(as_tuple=True)[0]
        if len(reset_env_ids) > 0:
            self._observation_buffer[reset_env_ids] = obs_kinematic_current[reset_env_ids].unsqueeze(1).repeat(
                1, self.cfg.observation_buffer_length, 1
            )
        
        # Flatten kinematics history channel: Shape (32768, 475)
        kinematic_history_flat = self._observation_buffer.view(self.num_envs, -1)

        # 4. Extract action history flat snapshot (50 elements, un-duplicated)
        obs_actions = torch.reshape(self._action_history, (self.num_envs, -1))

        # 5. Final Concat for Policy Input: Shape (32768, 525)
        obs_policy = torch.cat([kinematic_history_flat, obs_actions], dim=-1)

        # 6. Critic Setup (Asymmetric track retains full visibility)
        obs_privileged = torch.cat(
            [
                self._disturbance_force[:, 0, :],
                self._disturbance_moment[:, 0, :],
                self._push_time.unsqueeze(dim=1),
                self._push_duration.unsqueeze(dim=1),
                self._time_elapsed.unsqueeze(dim=1),
                self._current_impulse,
                self._actions_filtered,
                self._alpha,
            ],
            dim=-1,
        )
        obs_critic = torch.cat([obs_policy, obs_privileged], dim=-1)

        return {"policy": obs_policy, "critic": obs_critic}

    def _get_rewards(self) -> torch.Tensor:

        # determine if terminal state has been reached
        died, time_out = self._get_dones()
        too_fast_vel = self._stage_value(self.cfg.too_fast_vel, "too_fast_vel")
        
        # contacts
        current_contact_time = self.scene["contact_sensor"].data.current_contact_time[:, self._wheel_contact_ids]
        num_contact = torch.sum(current_contact_time > 0.0, dim=1)
        self._current_contacts = num_contact > 0
        new_contacts = torch.logical_and(torch.logical_xor(self._current_contacts , self._first_contact), self._current_contacts)
        new_contact_idx = torch.nonzero(new_contacts)
        self._first_contact[new_contact_idx] = new_contacts[new_contact_idx]

        # distance to goal
        distance_to_goal_xy = torch.linalg.norm(self._desired_pos_w[:, :2] - self._robot.data.root_link_pos_w[:, :2], dim=1)
        distance_to_goal_xy_mapped = torch.exp(-distance_to_goal_xy / 0.25)

        # height
        height = self._robot.data.root_link_pos_w[:, 2]
        height_mapped = torch.exp(-torch.square(height) / 0.25)

        # linear and angular velocity
        lin_vel = torch.sum(torch.square(self._robot.data.root_com_lin_vel_w), dim=1)
        lin_vel_mapped = torch.exp(-lin_vel / 0.25)
        ang_vel = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b[:, :2]), dim=1)
        ang_vel_mapped = torch.exp(-ang_vel / 0.25)
        spin_vel = torch.square(self._robot.data.root_com_ang_vel_b[:, 2])
        spin_vel_mapped = torch.exp(-spin_vel / 0.25)

        # reward descending z velocity
        descending_error = torch.square(self._robot.data.root_com_lin_vel_w[:, 2] - self._lin_vel_des[2]) * ~self._first_contact
        descending_error_mapped = torch.exp(-descending_error / 0.25)

        # tilt error
        tilt = self._robot.data.joint_pos[:, self._joint0]
        tilt_error = torch.square(tilt - 90 * pi / 180)
        tilt_error_mapped = torch.exp(-tilt_error / 0.25)

        # landing acceptance state
        self._in_acceptance_xy = (distance_to_goal_xy - self._stage_value(self.cfg.delta_d, "delta_d")) < 0.0

        # action rate (strictly flight thrusters to protect the arm)
        action_rate_flight = torch.sum(torch.square(self._actions[:, :4] - self._action_history[:, 0, :4]), dim=1)

        # reward low thruster actions that occur after first contact
        ground_thrust = torch.sum(torch.square(self._actions[:, :4]), dim=1) * self._first_contact
        ground_thrust_mapped = torch.exp(-ground_thrust / 0.25)

        # orientation
        flat_orientation = torch.abs(1 - self.quat_axis(self._robot.data.root_link_quat_w, 2)[..., 2])
        flat_orientation_mapped = torch.exp(-flat_orientation / 0.25)

        accepted_contact = self._current_contacts & ~died
        if self._stage_value(self.cfg.contact_reward_requires_xy_by_stage, "contact_reward_requires_xy_by_stage"):
            accepted_contact &= self._in_acceptance_xy
        new_valid_contact = new_contacts & accepted_contact
        has_landed = self._ep_contact_in_acceptance | accepted_contact
        self._ep_contact_in_acceptance = has_landed

        timeout_no_contact_penalty = (
            time_out.float()
            * (~has_landed).float()
            * self._stage_value(self.cfg.timeout_pen, "timeout_pen")
        )

        excess_vel = torch.clamp(torch.linalg.norm(self._robot.data.root_com_lin_vel_w, dim=1) - too_fast_vel, min=0.0)

        tuck_error = torch.square(tilt - (90 * pi / 180))
        tuck_multiplier = torch.exp(-tuck_error / 0.5) 

        # Now the drone gets paid proportionally to how well it tucked while landing
        gated_contact_rew = (
            accepted_contact.float()
            * tuck_multiplier
            * self._stage_value(self.cfg.contact_in_acceptance_rew_scale, "contact_in_acceptance_rew_scale")
            * self.step_dt
        )
        touchdown_rew = (
            new_valid_contact.float()
            * self._stage_value(self.cfg.touchdown_rew_scale, "touchdown_rew_scale")
            * (0.25 + 0.75 * tuck_multiplier)
        )

        rewards = {
            "lin_vel_pen": lin_vel * self._stage_value(self.cfg.lin_vel_pen_scale, "lin_vel_pen_scale") * self.step_dt,  
            "ang_vel_pen": ang_vel * self._stage_value(self.cfg.ang_vel_pen_scale, "ang_vel_pen_scale") * self.step_dt,  
            "action_rate_pen": action_rate_flight * self._stage_value(self.cfg.action_rate_pen_scale, "action_rate_pen_scale") * self.step_dt,  
            "ground_thrust_penalty": ground_thrust * self._stage_value(self.cfg.ground_thrust_pen_scale, "ground_thrust_pen_scale") * self.step_dt,  
            "spin_penalty": spin_vel * self._stage_value(self.cfg.spin_pen_scale, "spin_pen_scale") * self.step_dt,  
            "orientation_pen": flat_orientation * self._stage_value(self.cfg.orientation_pen_scale, "orientation_pen_scale") * self.step_dt,  
            
            "died_penalty": died * (self._stage_value(self.cfg.died_pen, "died_pen") - (excess_vel * 10.0)),                                                          
            
            "impulse_penalty": self._current_impulse.squeeze(dim=1) * self._stage_value(self.cfg.impulse_pen, "impulse_pen"),  
            
            "distance_to_goal_xy_rew": distance_to_goal_xy_mapped * self._stage_value(self.cfg.distance_to_goal_xy_rew_scale, "distance_to_goal_xy_rew_scale") * self.step_dt,  
            
            "descending_rew": descending_error_mapped * self._stage_value(self.cfg.descending_rew_scale, "descending_rew_scale") * self.step_dt,  
            
            "tilt_rew": tilt_error_mapped * self._stage_value(self.cfg.tilt_rew_scale, "tilt_rew_scale") * self.step_dt,  
            
            "touchdown_rew": touchdown_rew,
            "contact_in_acceptance_rew": gated_contact_rew,  
            "timeout_high_penalty": timeout_no_contact_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died1 = torch.linalg.norm(self._robot.data.root_com_lin_vel_w, dim=1) > self._stage_value(self.cfg.too_fast_vel, "too_fast_vel")
        
        # Origin-relative calculation ensures boundary tracking is pad-centric
        died2 = torch.linalg.norm(self._desired_pos_w[:, :2] - self._robot.data.root_link_pos_w[:, :2], dim=1) > self._stage_value(self.cfg.termination_dxy, "termination_dxy")
        died3 = self._robot.data.root_link_pos_w[:, 2] > self._stage_value(self.cfg.termination_height, "termination_height")
        
        bad_contact_ids = self._bad_contact_ids
        if self._stage_value(self.cfg.belly_contact_is_death_by_stage, "belly_contact_is_death_by_stage"):
            bad_contact_ids = self._belly_contact_ids + bad_contact_ids
        bad_contact_time = self.scene["contact_sensor"].data.current_contact_time[:, bad_contact_ids]
        died4 = torch.any(bad_contact_time > 0.0, dim=1)
        
        if self.cfg.terminate:
            died = died1 | died2 | died3 | died4
        else:
            died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_link_pos_w[env_ids], dim=1
        ).mean()
        
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        # 🛠️ FIXED: Restored self.reset_terminated and self.reset_time_outs attributes 
        # to clear the AttributeError while logging environment terminal steps.
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Curriculum/stage"] = int(self.cfg.curriculum_stage)
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/distance_to_goal_epoch_av"] = self.distance_to_goal_epoch_av
        extras["Metrics/final_height"] = self._robot.data.root_link_pos_w[env_ids, 2].mean().item()
        
        landed = self._ep_contact_in_acceptance[env_ids]
        timeout = self.reset_time_outs[env_ids]
        num_resets = max(len(env_ids), 1)

        extras["Landing/contact_acceptance_frequency"] = (
            torch.count_nonzero(landed & timeout).item() / num_resets
        )

        extras["Landing/timeout_with_contact_frequency"] = (
            torch.count_nonzero(timeout & landed).item() / num_resets
        )

        extras["Landing/timeout_no_contact_frequency"] = (
            torch.count_nonzero(timeout & ~landed).item() / num_resets
        )

        extras["Landing/terminated_no_contact_frequency"] = (
            torch.count_nonzero(self.reset_terminated[env_ids] & ~landed).item() / num_resets
        )
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self._is_first_sim_step and len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            self._is_first_sim_step = False
        
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            
        self._ep_contact_in_acceptance[env_ids] = False
        
        if self.cfg.randomize:
            self._actions[env_ids] = torch.ones_like(self._actions[env_ids])
            self._actions_filtered[env_ids] = torch.zeros_like(self._actions_filtered[env_ids])
            self._action_history[env_ids] = torch.zeros_like(self._action_history[env_ids])

            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-self.box_extent, self.box_extent)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])

            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]

            default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(self.cfg.initial_height_range[0], self.cfg.initial_height_range[1])
            default_root_state[:, 7:10] = torch.zeros_like(default_root_state[:, 7:10]).uniform_(self.cfg.initial_lin_vel_range[0], self.cfg.initial_lin_vel_range[1])
            default_root_state[:, 10:13] = torch.zeros_like(default_root_state[:, 10:13]).uniform_(self.cfg.initial_ang_vel_range[0], self.cfg.initial_ang_vel_range[1])
            default_root_state[:, 3:7], _ = self.random_quaternion(len(env_ids))

            joint_pos = self._robot.data.default_joint_pos[env_ids]
            random_tilt = torch.zeros_like(joint_pos[:, self._joint0]).uniform_(self.cfg.initial_tilt_range[0], self.cfg.initial_tilt_range[1])
            joint_pos[:, self._joint0] = random_tilt.clone()
            joint_pos[:, self._joint1] = random_tilt.clone()
            self._tilt_angle[env_ids, 0] = random_tilt.clone()

            joint_vel = self._robot.data.default_joint_vel[env_ids]
            random_vel = torch.zeros_like(joint_vel[:, self._joint0]).uniform_(self.cfg.initial_tilt_vel_range[0], self.cfg.initial_tilt_vel_range[1])
            joint_vel[:, self._joint0] = random_vel.clone()
            joint_vel[:, self._joint1] = random_vel.clone()

            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)

            self._first_contact[env_ids] = False
            self.kT[env_ids] = self.cfg.kT_0 * (1 + self.cfg.kT_error_scale * torch.zeros_like(self.kT[env_ids]).uniform_(-1., 1.))
            self.kM[env_ids] = self.cfg.kM_0 * (1 + self.cfg.kM_error_scale * torch.zeros_like(self.kM[env_ids]).uniform_(-1., 1.))
            self.max_tilt_vel[env_ids] = self.cfg.max_tilt_vel_0 * (1 + self.cfg.max_tilt_vel_error_scale * torch.zeros_like(self.max_tilt_vel[env_ids]).uniform_(-1., 1.))

            disturbance_force_direction = torch.normal(0.0, 1.0, size=(self.num_envs, 1, 3), device=self.device)
            disturbance_force_direction = disturbance_force_direction / (torch.linalg.norm(disturbance_force_direction, dim=1).unsqueeze(dim=1) + 1e-6)
            disturbance_moment_direction = torch.normal(0.0, 1.0, size=(self.num_envs, 1, 3), device=self.device)
            disturbance_moment_direction = disturbance_moment_direction / (torch.linalg.norm(disturbance_moment_direction, dim=1).unsqueeze(dim=1) + 1e-6)

            self._push_time[env_ids] = self.cfg.episode_length_s * torch.zeros_like(self._push_time[env_ids]).uniform_(0.0, 0.5)
            self._push_duration[env_ids] = torch.zeros_like(self._push_duration[env_ids]).uniform_(0.0, 0.2)

            force_intensity = torch.normal(torch.tensor(0.0), self.cfg.disturbance_force_scale)
            moment_intensity = torch.normal(torch.tensor(0.0), self.cfg.disturbance_moment_scale)
            self._disturbance_force = force_intensity * disturbance_force_direction
            self._disturbance_moment = moment_intensity * disturbance_moment_direction
            
            if self.cfg.randomize_motor_dynamics:
                self._alpha = torch.zeros_like(self._alpha).uniform_(self.cfg.alpha_range[0], self.cfg.alpha_range[1])

            self._time_elapsed[env_ids] = 0.0
            self._current_impulse[env_ids] = 0.0

            self._current_contacts[env_ids] = False
            self._in_acceptance_ball[env_ids] = False
            self._in_acceptance_xy[env_ids] = False

        else:
            self._actions[env_ids] = torch.zeros_like(self._actions[env_ids])
            self._actions_filtered[env_ids] = torch.zeros_like(self._actions_filtered[env_ids])
            self._action_history[env_ids] = torch.zeros_like(self._action_history[env_ids])

            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-0.1, 0.1)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])

            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]

            joint_pos = self._robot.data.default_joint_pos[env_ids]
            self._tilt_angle[env_ids, 0] = 0.0
            joint_vel = self._robot.data.default_joint_vel[env_ids]

            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)

            disturbance_force_direction = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1, 1)
            disturbance_force_direction = disturbance_force_direction / (torch.linalg.norm(disturbance_force_direction, dim=1).unsqueeze(dim=1) + 1e-6)
            disturbance_moment_direction = torch.normal(0.0, 1.0, size=(1, 1, 3), device=self.device).repeat(self.num_envs, 1, 1)
            disturbance_moment_direction = disturbance_moment_direction / (torch.linalg.norm(disturbance_moment_direction, dim=1).unsqueeze(dim=1) + 1e-6)

            self._push_time[env_ids] = 0.5 * torch.ones_like(self._push_time[env_ids])
            self._push_duration[env_ids] = 0.5 * torch.ones_like(self._push_duration[env_ids])

            force_intensity = 4 * self.cfg.kT_0 * 0.15
            moment_intensity = torch.normal(torch.tensor(0.0), self.cfg.disturbance_moment_scale)
            self._disturbance_force = force_intensity * disturbance_force_direction
            self._disturbance_moment = moment_intensity * disturbance_moment_direction

            self._first_contact[env_ids] = False
            self._time_elapsed[env_ids] = 0.0
            self._current_impulse[env_ids] = 0.0

            self._current_contacts[env_ids] = False
            self._in_acceptance_ball[env_ids] = False
            self._in_acceptance_xy[env_ids] = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def random_quaternion(self, num):
        roll = torch.pi / 6 * (2 * torch.rand(num, dtype=torch.float) - 1)
        pitch = torch.pi / 6 * (2 * torch.rand(num, dtype=torch.float) - 1)
        yaw = 2 * torch.pi * torch.rand(num, dtype=torch.float)
        return quat_from_euler_xyz(roll, pitch, yaw), yaw.to(self.device)

    def quat_axis(self, q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)

    def tensor_clamp(self, t, min_t, max_t):
        return torch.max(torch.min(t, max_t), min_t)
