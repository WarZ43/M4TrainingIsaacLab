# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from numpy import pi, exp

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_rotate
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor

##
# Pre-defined configs
##
from omni.isaac.lab_assets import ATMO_CFG as ATMO_ROBOT_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

from .landing_task import LandingTask
from .randomizations import M4Randomizer
from .vehicle_adapters import GenericM4VehicleAdapter
from .vehicle_specs import ATMO_SPEC


class M4EnvWindow(BaseEnvWindow):
    """Window manager for the M4 landing environment."""

    def __init__(self, env: M4Env, window_name: str = "IsaacLab"):
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
class M4EnvCfg(DirectRLEnvCfg):
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
    contact_reward_requires_xy_by_stage = [True, False]

    curriculum_update_rate = 8e3
    curriculum_steps_to_completion = curriculum_update_rate * 10

    # action history
    action_history_length = 25
    observation_history_length = 24

    # env
    episode_length_s = 5.0
    sim_dt = 1 / 50  # training 1/100
    decimation = 1  # training 2
    action_space = ATMO_SPEC.action_dim

    num_obs = ATMO_SPEC.observation_dim
    observation_space = ((observation_history_length + 1) * num_obs) + (action_space * action_history_length)

    num_privileged_obs = 7 + action_space
    state_space = 0
    debug_vis = True
    num_envs = 32768

    ui_window_class_type = M4EnvWindow

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
    robot: ArticulationCfg = ATMO_ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", track_air_time=True, history_length=2
    )

    # termination conditions
    too_fast_vel = [2.0, 2.0]
    termination_dxy = [1.50, 3.50]
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

    distance_to_goal_xy_rew_scale = [0.30, 0.10]
    descending_rew_scale = [0.30, 0.60]
    tilt_rew_scale = [0.80, 0.80]
    contact_in_acceptance_rew_scale = [0.40, 4.00]
    touchdown_rew_scale = [0.00, 3.00]

    # nominal parameters
    kT_0 = 28.15
    kM_0 = 0.018
    max_tilt_vel_0 = pi / 8

    # random force and torque scales
    disturbance_force_scale = 4 * kT_0 * 0.50  # best 0.05
    disturbance_moment_scale = 4 * kT_0 * kM_0 * 0.05  # best 0.05

    dist_force_cts_scale = 4 * kT_0 * 0.0
    dist_moment_cts_scale = 4 * kT_0 * kM_0 * 0.0

    # randomization parameters
    kT_error_scale = 0.2
    kM_error_scale = 0.2
    # Scalar or stage-indexed maximum per-rotor thrust loss. VehicleSpec decides the start stage.
    thrust_loss_max = 0.2
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
    pos_noise_scale = 0.005  # 0.5 cm
    quat_noise_scale = 0.005  # 0.5 percent
    lin_vel_noise_scale = 0.035  # 0.035 m/s
    ang_vel_noise_scale = 0.035  # 2 deg/s or 0.035 rad/s
    tilt_noise_scale = 0.018  # 2 degrees or 0.18 rad/s
    roll_noise_scale = 0.008  # 2 degrees or 0.18 rad/s
    pitch_noise_scale = 0.008  # 2 degrees or 0.18 rad/s
    rot_noise_scale = 0.005  # 0.5 percent


class M4Env(DirectRLEnv):
    cfg: M4EnvCfg

    def __init__(self, cfg: M4EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.box_extent = 0.1
        self.curriculum_update_time = 0
        self.distance_to_goal_epoch_av = 0.0

        self.vehicle = GenericM4VehicleAdapter(self, ATMO_SPEC)
        self.vehicle.initialize()
        self.task = LandingTask(self)
        self.randomizer = M4Randomizer(self)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self._is_first_sim_step = True

    def _stage_value(self, values, name: str):
        return self.task.stage_value(values, name)

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
        self.vehicle.pre_physics_step(actions)

    def _apply_action(self):
        self.vehicle.apply_action()

    def _get_observations(self) -> dict:
        return self.vehicle.get_observations()

    def _get_rewards(self) -> torch.Tensor:
        return self.task.get_rewards()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.task.get_dones()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self.task.log_episode(env_ids)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self._is_first_sim_step and len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            self._is_first_sim_step = False

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.randomizer.reset(env_ids)
        self.task.reset_episode_state(env_ids)

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
