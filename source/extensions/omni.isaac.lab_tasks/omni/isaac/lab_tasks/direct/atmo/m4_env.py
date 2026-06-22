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

from .landing_task import LandingTask, LandingTaskCfg
from .randomizations import M4Randomizer
from .takeoff_task import TakeoffTask, TakeoffTaskCfg
from .vehicle_adapters import GenericM4VehicleAdapter
from .vehicle_specs import ATMO_SPEC


class M4EnvWindow(BaseEnvWindow):
    """Window manager for the M4 task environment."""

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
    task_name = "landing"
    seed = 42

    # curriculum stage selection: 1 = base-frame quadcopter thrust, 2 = per-rotor thrust vectoring
    curriculum_stage = 2
    rotate_rotor_thrust_by_stage = [False, True]
    rotate_rotor_thrust_by_body_by_stage = [True, False]
    collective_vertical_thrust_by_stage = [False, False]
    collapse_rotor_wrench_to_base_by_stage = [False, False]
    apply_rotor_moments_by_stage = [True, True]
    rotor_moment_scale_by_stage = [0.1, 1.0]
    ramp_steps_per_epoch = 24
    thrust_rotation_ramp_start_epoch = 0.0
    thrust_rotation_ramp_end_epoch = 0.0
    disturbance_ramp_start_epoch = 0.0
    disturbance_ramp_end_epoch = 0.0

    curriculum_update_rate = 8e3
    curriculum_steps_to_completion = curriculum_update_rate * 10
    landing: LandingTaskCfg = LandingTaskCfg()
    takeoff: TakeoffTaskCfg = TakeoffTaskCfg()

    # action history
    action_history_length = 25
    observation_history_length = 25

    # env
    episode_length_s = 5.0
    sim_dt = 1 / 50  # training 1/100
    decimation = 1  # training 2
    action_space = ATMO_SPEC.action_dim

    num_obs = ATMO_SPEC.observation_dim
    num_current_obs = ATMO_SPEC.current_observation_dim
    observation_space = (
        (observation_history_length * num_obs) + (action_space * action_history_length) + num_current_obs
    )

    num_privileged_obs = 7 + action_space
    state_space = 0
    debug_vis = True
    thrust_vector_debug_vis = True
    thrust_vector_debug_num_envs = 64
    thrust_vector_debug_scale = 0.02
    disturbance_force_debug_scale = 0.04
    disturbance_moment_debug_scale = 0.25
    thrust_vector_debug_line_width = 3.0
    thrust_vector_debug_head_length = 0.08
    thrust_vector_debug_head_width = 0.035
    num_envs = 32768

    ui_window_class_type = M4EnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        disable_contact_processing=True,
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=2**19,
            gpu_max_rigid_contact_count=2**24,
        ),
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

    # observation delay (num steps)
    observation_buffer_length = observation_history_length
    observation_delay = 0

    # nominal parameters
    kT_0 = 28.15
    kM_0 = 0.018
    max_tilt_vel_0 = pi / 8

    # random force and torque scales
    disturbance_force_scale = 4 * kT_0 * 0.05
    disturbance_moment_scale = 4 * kT_0 * kM_0 * 0.005

    dist_force_cts_scale = 4 * kT_0 * 0.0
    dist_moment_cts_scale = 4 * kT_0 * kM_0 * 0.0

    # randomization parameters
    kT_error_scale = 0.2
    kM_error_scale = 0.2
    # Scalar or stage-indexed maximum per-rotor thrust loss. VehicleSpec decides the start stage.
    thrust_loss_max = [0.0, 0.2]
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
        self._global_env_step = 0
        self._debug_draw = None

        self.vehicle = GenericM4VehicleAdapter(self, ATMO_SPEC)
        self.vehicle.initialize()
        self.task = self._make_task()
        self.randomizer = M4Randomizer(self)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self._is_first_sim_step = True

    def _stage_value(self, values, name: str):
        return self.task.stage_value(values, name)

    def _training_epoch(self) -> float:
        return float(self._global_env_step) / max(float(self.cfg.ramp_steps_per_epoch), 1.0)

    def _epoch_ramp(self, start_epoch: float, end_epoch: float) -> float:
        if end_epoch <= start_epoch:
            return 1.0
        progress = (self._training_epoch() - float(start_epoch)) / (float(end_epoch) - float(start_epoch))
        return max(0.0, min(1.0, progress))

    def thrust_rotation_weight(self) -> float:
        return self._epoch_ramp(
            self.cfg.thrust_rotation_ramp_start_epoch,
            self.cfg.thrust_rotation_ramp_end_epoch,
        )

    def disturbance_weight(self) -> float:
        return self._epoch_ramp(
            self.cfg.disturbance_ramp_start_epoch,
            self.cfg.disturbance_ramp_end_epoch,
        )

    def _make_task(self):
        task_name = self.cfg.task_name.lower()
        if task_name == "landing":
            return LandingTask(self, self.cfg.landing)
        if task_name == "takeoff":
            return TakeoffTask(self, self.cfg.takeoff)
        raise ValueError(f"Unsupported task_name '{self.cfg.task_name}'. Expected 'landing' or 'takeoff'.")

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
        self._global_env_step += 1
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
            if self._debug_draw is None:
                try:
                    from omni.isaac.debug_draw import _debug_draw

                    self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                except Exception:
                    self._debug_draw = None
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if self._debug_draw is not None:
                self._debug_draw.clear_lines()

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        self._draw_force_debug_vectors()

    def _draw_force_debug_vectors(self):
        if not self.cfg.thrust_vector_debug_vis or self._debug_draw is None:
            return
        if not hasattr(self, "_debug_thrust_w") or not self.vehicle.rotor_ids:
            return

        num_envs = min(int(self.cfg.thrust_vector_debug_num_envs), self.num_envs)
        if num_envs <= 0:
            self._debug_draw.clear_lines()
            return

        body_pos_w = getattr(self._robot.data, "body_pos_w", None)
        if body_pos_w is None:
            body_pos_w = self._robot.data.body_state_w[..., :3]
        rotor_pos_w = body_pos_w[:num_envs, self.vehicle.rotor_ids, :]
        thrust_w = self._debug_thrust_w[:num_envs]
        moment_w = self._debug_moment_w[:num_envs]
        starts = []
        ends = []
        colors = []
        widths = []

        thrust_starts = rotor_pos_w.reshape(-1, 3)
        thrust_ends = (rotor_pos_w + self.cfg.thrust_vector_debug_scale * thrust_w).reshape(-1, 3)
        moment_z = moment_w.reshape(-1, 3)[:, 2]
        thrust_colors = [self._moment_debug_color(moment_value) for moment_value in moment_z.detach().cpu().tolist()]
        self._append_debug_arrows(thrust_starts, thrust_ends, thrust_colors, starts, ends, colors, widths)

        if hasattr(self, "_debug_disturbance_force_w"):
            base_pos_w = body_pos_w[:num_envs, self.vehicle.base_link, :].unsqueeze(1)
            dist_force_w = self._debug_disturbance_force_w[:num_envs]
            force_starts = base_pos_w.reshape(-1, 3)
            force_ends = (base_pos_w + self.cfg.disturbance_force_debug_scale * dist_force_w).reshape(-1, 3)
            force_colors = [(1.0, 0.2, 1.0, 1.0)] * force_starts.shape[0]
            self._append_debug_arrows(force_starts, force_ends, force_colors, starts, ends, colors, widths)

        if hasattr(self, "_debug_disturbance_moment_w"):
            base_pos_w = body_pos_w[:num_envs, self.vehicle.base_link, :].unsqueeze(1)
            dist_moment_w = self._debug_disturbance_moment_w[:num_envs]
            moment_starts = (base_pos_w + torch.tensor([0.0, 0.0, 0.12], device=self.device)).reshape(-1, 3)
            moment_ends = (
                base_pos_w
                + torch.tensor([0.0, 0.0, 0.12], device=self.device)
                + self.cfg.disturbance_moment_debug_scale * dist_moment_w
            ).reshape(-1, 3)
            moment_colors = [(1.0, 1.0, 1.0, 1.0)] * moment_starts.shape[0]
            self._append_debug_arrows(moment_starts, moment_ends, moment_colors, starts, ends, colors, widths)

        self._debug_draw.clear_lines()
        if not starts:
            return
        self._debug_draw.draw_lines(starts, ends, colors, widths)

    def _append_debug_arrows(
        self,
        arrow_starts: torch.Tensor,
        arrow_ends: torch.Tensor,
        arrow_colors: list[tuple[float, float, float, float]],
        starts: list[list[float]],
        ends: list[list[float]],
        colors: list[tuple[float, float, float, float]],
        widths: list[float],
    ):
        vectors = arrow_ends - arrow_starts
        lengths = torch.linalg.norm(vectors, dim=1, keepdim=True)
        valid = lengths.squeeze(dim=1) > 1e-6
        starts.extend(arrow_starts.detach().cpu().tolist())
        ends.extend(arrow_ends.detach().cpu().tolist())
        colors.extend(arrow_colors)
        widths.extend([float(self.cfg.thrust_vector_debug_line_width)] * arrow_starts.shape[0])
        if torch.any(valid):
            directions = vectors[valid] / lengths[valid].clamp_min(1e-6)
            up = torch.zeros_like(directions)
            up[:, 2] = 1.0
            nearly_vertical = torch.abs(directions[:, 2]) > 0.95
            up[nearly_vertical] = torch.tensor([1.0, 0.0, 0.0], device=up.device)
            side = torch.cross(directions, up, dim=1)
            side = side / torch.linalg.norm(side, dim=1, keepdim=True).clamp_min(1e-6)

            head_length = torch.clamp(
                0.25 * lengths[valid],
                max=float(self.cfg.thrust_vector_debug_head_length),
            )
            head_width = torch.clamp(
                0.12 * lengths[valid],
                max=float(self.cfg.thrust_vector_debug_head_width),
            )
            tip = arrow_ends[valid]
            head_center = tip - head_length * directions
            left = head_center + head_width * side
            right = head_center - head_width * side

            valid_colors = [color for color, keep in zip(arrow_colors, valid.detach().cpu().tolist()) if keep]
            starts.extend(tip.detach().cpu().tolist())
            ends.extend(left.detach().cpu().tolist())
            colors.extend(valid_colors)
            widths.extend([float(self.cfg.thrust_vector_debug_line_width)] * len(valid_colors))
            starts.extend(tip.detach().cpu().tolist())
            ends.extend(right.detach().cpu().tolist())
            colors.extend(valid_colors)
            widths.extend([float(self.cfg.thrust_vector_debug_line_width)] * len(valid_colors))

    @staticmethod
    def _moment_debug_color(moment_z: float) -> tuple[float, float, float, float]:
        if moment_z > 1e-6:
            return (0.0, 0.75, 1.0, 1.0)
        if moment_z < -1e-6:
            return (1.0, 0.35, 0.0, 1.0)
        return (0.7, 0.7, 0.7, 1.0)

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
