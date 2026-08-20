# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from math import cos, exp, sin

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_rotate
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor

##
# Pre-defined configs
##
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

from .landing_task import LandingTask, LandingTaskCfg
from .combined_task import CombinedTask, CombinedTaskCfg
from .m4_config import MODE, RAMP_EPOCH_OFFSET, STAGE_RAMP_START_EPOCH, TASK
from .randomizations import M4Randomizer
from .takeoff_task import TakeoffTask, TakeoffTaskCfg
from .base import BaseTask
from .vehicle_adapters import DroneRuntime
from .vehicle_specs import VehicleSpec
from .vehicle_profiles import ACTIVE_PROFILE, GROUND_MATERIAL, M4VehicleProfile

_PROFILE = ACTIVE_PROFILE
_RUN_MODE = os.environ.get("M4_RUN_MODE", MODE).lower()


def _episode_length_for_task(task_name: str) -> float:
    task_name = task_name.lower()
    if task_name == "takeoff":
        return 10.0
    if task_name == "landing":
        return 8.0
    if task_name == "combined":
        return 12.0
    raise ValueError(f"Unsupported task_name '{task_name}'. Expected 'landing', 'takeoff', or 'combined'.")


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def _env_nonnegative_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0, got {parsed}")
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


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
    rl_games_name = _PROFILE.rl_games_name()
    reuse_last_stage_value = _PROFILE.reuse_last_stage_value

    # high level flags
    randomize = True
    terminate = True
    disturb = True
    noise = True
    randomize_motor_dynamics = True
    # Center-of-mass randomization (2026-08-19). MEASURED, not guessed: the USD
    # asset places the COM at ~6.0 mm and the MPC model at 3.3 mm, so the model
    # disagreement the policy has to absorb is ~2.7 mm. The band is set to span
    # that discrepancy in both directions rather than to some invented width --
    # an earlier +-20 mm draft was ~7x the real uncertainty and would have
    # spent the campaign training against a plant that does not exist.
    # NOTE: the 6.0 / 3.3 mm figures are a single axis; applied to all three
    # here because the axis was not recorded. Narrow to the measured axis once
    # known -- randomizing an axis with no real uncertainty only costs sample
    # efficiency.
    randomize_com = True
    # MEASURED 2026-08-19 (hang test + balance): the VEHICLE CG sits ~6.0 mm
    # forward of the rotor center. The USD puts it at 2.90 mm, so the model is
    # ~2x short and the correction is +3.1 mm of vehicle CG, FORWARD.
    #
    # UNITS TRAP: these offsets are applied to BASE_LINK's own COM, and
    # base_link is 2.33 kg of the vehicle's 5.488 kg (42.5%). A base_link
    # offset therefore moves the VEHICLE CG by only 0.425x as much. Converting:
    #     nominal -7.30 mm on base_link  ->  +3.10 mm vehicle CG  ->  6.0 mm total
    #     band    +-4.70 mm on base_link ->  +-2.00 mm vehicle CG
    # Forward is body -x (ATMO_SPEC.forward_yaw_offset = pi), hence the sign.
    #
    # Vertical is deliberately NOT corrected here. The same measurement puts the
    # real CG 15.7 mm ABOVE the USD's, but that is unmodeled mass mounted high
    # and forward, not a COM offset -- fix the mass distribution, not this knob.
    com_nominal_offset_b = (-0.00730, 0.0, 0.0)   # m, base_link body frame
    com_offset_range_xy = (-0.0047, 0.0047)   # m, +-4.7 mm -> +-2.0 mm vehicle CG
    com_offset_range_z = (-0.0047, 0.0047)    # m, same band
    quantize_tilt_action = True
    # Deployment parity: the sim and hardware runtimes zero rotor thrust while
    # the reference mode is drive and zero the wheel speed target while airborne
    # (takeoff or flight). Train under the same gating so the deployed policy
    # never meets an actuation boundary it did not experience.
    drive_zero_thrust = _env_bool("M4_DRIVE_ZERO_THRUST", True)
    wheels_ground_only = _env_bool("M4_WHEELS_GROUND_ONLY", True)
    task_name = TASK
    combined_sequence_test = _env_bool("M4_COMBINED_SEQUENCE_TEST", False)
    seed = 42

    # Curriculum stage selection:
    # 1 = real thrust-vector dynamics with common morph only.
    # 2 = unlock morph tilt asymmetry.
    # 3 = unlock thrust-center trim and thrust-loss ramp.
    curriculum_stage = _PROFILE.curriculum_stage
    morph_bias_start_stage = 2
    thrust_center_start_stage = 3
    morph_balance_authority_ramp_epochs = 50.0
    thrust_center_authority_ramp_epochs = 50.0
    new_dof_randomization_ramp_epochs = 50.0
    wheel_action_start_stage = 1
    thrust_loss_start_stage = 2
    is_final_stage = _PROFILE.final_stage_enabled()
    final_stage_task_names = ("landing", "takeoff", "combined")
    final_stage_start_epoch = _PROFILE.final_stage_epoch()
    final_stage_overlay_stage = _PROFILE.final_overlay_stage()
    final_stage_thrust_loss_stage = _PROFILE.final_thrust_loss_stage()
    # Stage-3 hardening overlay. The overlay above lifts stages 1-2 onto
    # stage-3 values and is inert once curriculum_stage is already 3; this one is
    # its complement and fires only at stage 3, after the policy has trained
    # there long enough to be tracking the reference. It trades reference
    # fidelity for hardware robustness: the gross-deviation termination is
    # lifted outright, and the actuator and initial-state randomization bands
    # widen. The widening is ramped over final_stage3_ramp_epochs rather than
    # stepped -- a converged stage-3 policy meeting the full band in one epoch
    # loses more than the extra robustness is worth.
    final_stage3_overlay_enabled = _PROFILE.final_stage3_enabled()
    final_stage3_start_epoch = _PROFILE.final_stage3_epoch()
    final_stage3_ramp_epochs = 100.0
    ramp_steps_per_epoch = 36
    ramp_epoch_offset = RAMP_EPOCH_OFFSET
    stage_ramp_start_epoch = STAGE_RAMP_START_EPOCH
    # Stationary easy/moderate/hard disturbance mix. The bucket scale is applied
    # to force/moment disturbances and thrust loss, not to other randomizations.
    disturbance_batch_probabilities = (
        (0.30, 0.30, 0.40),
        (0.20, 0.40, 0.40),
        (0.50, 0.25, 0.25),
    )
    disturbance_batch_scale_multipliers = (0.0, 0.5, 1.0)
    thrust_loss_batch_scale_multipliers = (0.0, 0.0, 1.0)
    # Early-training disturbance ramp. This machinery already existed but was a
    # NO-OP: initial_scale 1.0 with start == end == 0 makes _epoch_ramp return
    # 1.0, so stage 1 ran at full disturbance from epoch 0.
    #
    # That is a lot to hand a policy that has not found the behavior yet. The
    # stage-1 bucket mix is (0.30, 0.30, 0.40) against multipliers
    # (0.0, 0.5, 1.0), so 40% of envs saw the full 3.35 scale immediately, on an
    # airframe with a thrust-to-weight of about 2.0.
    #
    # Stage 1 now starts at a quarter strength and reaches full by epoch 200 --
    # after takeoff discovery (measured at epoch 100-108 across two runs) and
    # before the aggressive training_overlay at 300. Stages 2 and 3 are
    # unchanged: they inherit a policy that already works.
    disturbance_ramp_initial_scale_by_stage = (0.25, 1.0, 1.0)
    disturbance_ramp_start_epoch_by_stage = (0.0, 0.0, 0.0)
    disturbance_ramp_end_epoch_by_stage = (200.0, 0.0, 0.0)

    landing: LandingTaskCfg = _PROFILE.landing
    takeoff: TakeoffTaskCfg = _PROFILE.takeoff
    combined: CombinedTaskCfg = _PROFILE.combined

    # action history
    action_history_length = _env_positive_int("M4_ACTION_HISTORY_LENGTH", 25)
    observation_history_length = _env_positive_int("M4_OBSERVATION_HISTORY_LENGTH", 15)
    observation_delay_min_steps = _env_nonnegative_int("M4_OBSERVATION_DELAY_MIN_STEPS", 0)
    # Hardware transport can exceed one policy step; the previous max was 1.
    observation_delay_max_steps = _env_nonnegative_int("M4_OBSERVATION_DELAY_MAX_STEPS", 2)
    if observation_delay_max_steps < observation_delay_min_steps:
        raise ValueError(
            "M4_OBSERVATION_DELAY_MAX_STEPS must be >= M4_OBSERVATION_DELAY_MIN_STEPS, "
            f"got {observation_delay_max_steps} < {observation_delay_min_steps}"
        )

    # env
    # Takeoff needs enough horizon to climb and hold a steady hover.
    episode_length_s = _episode_length_for_task(task_name)
    sim_dt = 1 / 50  # training 1/100
    decimation = 1  # training 2
    action_space = _PROFILE.spec.action_dim

    num_obs = _PROFILE.spec.observation_dim
    num_current_obs = _PROFILE.spec.current_observation_dim
    # Mode one-hot (4) + signed phase-event countdown (1). Both are commanded
    # quantities; estimator-derived vectors live in the heading-frame terms.
    task_observation_dim = 5 if task_name.lower() == "combined" else 0
    observation_space = (
        (observation_history_length * num_obs)
        + (action_space * action_history_length)
        + num_current_obs
        + task_observation_dim
    )

    state_space = 0
    debug_vis = _RUN_MODE == "play"
    thrust_vector_debug_vis = _RUN_MODE == "play"
    thrust_vector_debug_num_envs = 64
    thrust_vector_debug_scale = 0.02
    disturbance_force_debug_scale = 0.04
    disturbance_moment_debug_scale = 0.25
    thrust_vector_debug_line_width = 3.0
    thrust_vector_debug_head_length = 0.08
    thrust_vector_debug_head_width = 0.035
    wheel_command_debug_vis = False
    wheel_command_debug_scale = 0.35
    wheel_command_debug_z_offset = 0.18
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
        physics_material=GROUND_MATERIAL,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = _PROFILE.robot_cfg.replace(prim_path=_PROFILE.robot_prim_path)

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", track_air_time=True, history_length=2
    )

    # observation delay (num steps)
    # One current frame, the 15-frame policy stack, and the maximum delay margin.
    observation_buffer_length = observation_history_length + observation_delay_max_steps + 1

    # random force and torque scale coefficients, relative to vehicle nominal thrust / rotor moment capacity
    disturbance_force_scale = _PROFILE.disturbance_force_scale
    disturbance_moment_scale = _PROFILE.disturbance_moment_scale

    dist_force_cts_scale = _PROFILE.dist_force_cts_scale
    dist_moment_cts_scale = _PROFILE.dist_moment_cts_scale

    # randomization parameters
    # Scalar or stage-indexed maximum per-rotor thrust loss. VehicleSpec decides the start stage.
    thrust_loss_max = _PROFILE.thrust_loss_max
    final_thrust_gain_range = (0.75, 1.2)
    # Stage-3 overlay targets for the four randomization axes it widens. Each is
    # interpolated from the nominal band above toward these values across the
    # overlay ramp; they are the endpoints, not a separate regime.
    # Per-vehicle: it must clear that vehicle's own stage-3 thrust_loss_max or it
    # is a no-op. ATMO 0.12 -> 0.20, m4tii 0.20 -> 0.28.
    final_stage3_thrust_loss_max = _PROFILE.final_stage3_thrust_loss()
    final_stage3_thrust_gain_range = (0.65, 1.30)
    # Latency. Widened in both directions around the 0.150 s nominal T_m_0.
    # Narrowed 2026-08-13 from (0.05, 0.35): the real tau is confidently inside
    # 0.1-0.2, and the observed-tau channel makes hedging beyond that wasteful.
    final_stage3_actuator_tau_range = (0.10, 0.20)
    # Multiplier on every reset position/attitude/velocity error envelope.
    final_stage3_initial_error_multiplier = 1.75
    minimum_thrust_scale = 0.65
    initial_lin_vel_range = [-0.1, 0.1]
    initial_ang_vel_range = [-0.1, 0.1]
    initial_tilt_range = _PROFILE.initial_tilt_range
    initial_leg_range = _PROFILE.initial_leg_range
    initial_tilt_vel_range = _PROFILE.initial_tilt_vel_range

    # low pass filter constant
    step_dt = sim_dt * decimation
    # Hardware-readiness: the deployed actuator lag is the software filter plus
    # transport plus the motor model, and the hardware share is uncertain, so
    # train across a band wide enough to cover both directions. The previous
    # range was [0.125, 0.175].
    actuator_tau_min_s = _env_float("M4_ACTUATOR_TAU_MIN_S", 0.10)
    actuator_tau_max_s = _env_float("M4_ACTUATOR_TAU_MAX_S", 0.20)
    T_m_0 = 0.150
    alpha_0 = 1.0 - exp(-step_dt / T_m_0)

    # ---- per-joint servo dynamics (Phase 3) --------------------------------
    # Measured on the vehicle 2026-08-11. Each joint draws INDEPENDENTLY: the
    # census found the four hips diverging 0.35 rad within seconds of enabling,
    # which a single shared draw cannot reproduce.
    #
    # STAGE-AWARE. Asymmetry between joints in a group is only correctable
    # through the action that group's balance channel provides -- tilt_balance
    # for the hips, thrust_center_xy for the legs -- and both are held at zero
    # by action_authority until their start stage. So each quantity is drawn as
    # a SHARED component (all stages) plus a PER-JOINT deviation scaled by that
    # authority. Stage 1 gets the full magnitude of every range but zero spread;
    # the spread ramps in exactly as the actuator that answers it unlocks.
    randomize_joint_dynamics = True
    # pi/8 = 0.3927 is the hips' profile-velocity cap and sits inside this band.
    # AT the cap the servo has no headroom to close an error -- the session's
    # worst tracking error (0.067 rad, two thirds of the 0.1 rad safety leash)
    # appeared exactly there.
    joint_velocity_ceiling_range = (0.30, 0.45)
    # Servo zero offsets. Legs are asymmetric because they sag under gravity:
    # the census found rear legs resting 0.22-0.39 rad extended while the fronts
    # sat within 0.06 of zero.
    hip_zero_offset_range = (-0.06, 0.06)
    leg_zero_offset_range = (-0.10, 0.25)

    # ---- state estimation noise (Phase 4) ----------------------------------
    # PROVISIONAL -- not measured on this vehicle, OptiTrack has been
    # unavailable. Revise during stage C. Each episode draws its own scale from
    # the band rather than running at a fixed value, so the policy cannot tune
    # itself to one noise level.
    randomize_observation_noise = True
    pos_noise_scale_range = (0.002, 0.010)  # was fixed 0.005
    # Upper end capped at ~1.3x nominal (was 0.06 = 1.7x): velocity feedback is
    # the derivative channel, and pricing it as maximally untrustworthy trains
    # a pure-P policy. Ranges remain PROVISIONAL (unmeasured).
    lin_vel_noise_scale_range = (0.02, 0.045)  # was fixed 0.035

    # observation noise scales (nominal / fallback when randomization is off)
    pos_noise_scale = 0.005  # 0.5 cm
    quat_noise_scale = 0.005  # 0.5 percent
    lin_vel_noise_scale = 0.035  # 0.035 m/s
    ang_vel_noise_scale = 0.035  # 2 deg/s or 0.035 rad/s
    tilt_noise_scale = 0.018  # 2 degrees or 0.18 rad/s
    roll_noise_scale = 0.008  # 2 degrees or 0.18 rad/s
    pitch_noise_scale = 0.008  # 2 degrees or 0.18 rad/s
    rot_noise_scale = 0.005  # 0.5 percent

    # ---- resilience beyond randomization (Phase 5) -------------------------
    # Widening ranges is not the same as teaching recovery.
    #
    # Command dropout. The deployment backend zeroes rotors after 100 ms and
    # holds joint targets; the policy should not be surprised by its own stream
    # stuttering. Hold the last action for a sampled window.
    # 0.015/step: with holds of 0.10-0.25 s (5-13 steps at training step_dt
    # 0.02 s) the expected duty cycle is ~11% of steps held. The original 0.05
    # worked out to ~30% of all steps with the policy's output ignored, which
    # for a fresh stage-1 policy is a credit-assignment tax, not resilience.
    command_dropout_probability = 0.015
    command_dropout_hold_s = (0.10, 0.25)
    # Saturated collective. The policy currently pins rotors at 1.000 whenever
    # altitude error exceeds ~1 m and has no experience of asking for more
    # thrust than it can get. Cap the collective for a whole episode.


class M4Env(DirectRLEnv):
    cfg: M4EnvCfg

    def __init__(
        self,
        cfg: M4EnvCfg,
        render_mode: str | None = None,
        *,
        vehicle_spec: VehicleSpec | None = None,
        task: BaseTask | None = None,
        **kwargs,
    ):
        cfg.episode_length_s = _episode_length_for_task(cfg.task_name)
        super().__init__(cfg, render_mode, **kwargs)

        self.box_extent = 0.1
        self.curriculum_update_time = 0
        self.distance_to_goal_epoch_av = 0.0
        self._global_env_step = 0
        self._debug_draw = None
        self._stage_value_cache = {}
        self._training_epoch_cache_step = -1
        self._training_epoch_cache_value = 0.0
        self._final_stage_overlay_cache_step = -1
        self._final_stage_overlay_cache_value = False
        self._final_stage3_overlay_cache_step = -1
        self._final_stage3_overlay_cache_value = False
        self._disturbance_weight_cache_step = -1
        self._disturbance_weight_cache_value = 0.0
        self.vehicle = DroneRuntime(self, vehicle_spec or _PROFILE.spec)
        self.vehicle.initialize()
        self.task = task or self._make_task()
        self.vehicle.bind_task(self.task)
        self.randomizer = M4Randomizer(self)
        self.randomizer.bind_task(self.task)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self._is_first_sim_step = True
        self._cached_dones_step = -1
        self._cached_dones: tuple[torch.Tensor, torch.Tensor] | None = None

    def _stage_value(self, values, name: str):
        stage = self.curriculum_stage_for_value(name)
        cache_key = (id(values), name, stage)
        cached = self._stage_value_cache.get(cache_key)
        if cached is not None:
            return cached
        value = self.task.stage_value(values, name)
        self._stage_value_cache[cache_key] = value
        return value

    def final_stage_overlay_active(self) -> bool:
        if self._final_stage_overlay_cache_step == self._global_env_step:
            return self._final_stage_overlay_cache_value
        final_stage_task_names = tuple(
            str(name).lower() for name in getattr(self.cfg, "final_stage_task_names", ("landing",))
        )
        active = (
            bool(getattr(self.cfg, "is_final_stage", False))
            and self.cfg.task_name.lower() in final_stage_task_names
            and int(self.cfg.curriculum_stage) != 3
            and self._training_epoch() >= float(getattr(self.cfg, "final_stage_start_epoch", 200.0))
        )
        self._final_stage_overlay_cache_step = self._global_env_step
        self._final_stage_overlay_cache_value = active
        return active

    def final_stage3_overlay_active(self) -> bool:
        """Stage-3 hardening overlay gate.

        Deliberately the complement of ``final_stage_overlay_active``: that one
        requires ``curriculum_stage != 3`` because its job is to lift the earlier
        stages onto stage-3 values, so it can never fire for a policy that is
        already training at stage 3. This one requires ``== 3``. The two are
        mutually exclusive by construction and never compose.
        """
        if self._final_stage3_overlay_cache_step == self._global_env_step:
            return self._final_stage3_overlay_cache_value
        final_stage_task_names = tuple(
            str(name).lower() for name in getattr(self.cfg, "final_stage_task_names", ("landing",))
        )
        active = (
            bool(getattr(self.cfg, "final_stage3_overlay_enabled", False))
            and self.cfg.task_name.lower() in final_stage_task_names
            and int(self.cfg.curriculum_stage) == 3
            and self._training_epoch() >= float(getattr(self.cfg, "final_stage3_start_epoch", 400.0))
        )
        self._final_stage3_overlay_cache_step = self._global_env_step
        self._final_stage3_overlay_cache_value = active
        return active

    def final_stage3_progress(self) -> float:
        """0 while the overlay is inactive, ramping to 1 across the overlay ramp."""
        if not self.final_stage3_overlay_active():
            return 0.0
        start_epoch = float(self.cfg.final_stage3_start_epoch)
        ramp_epochs = float(getattr(self.cfg, "final_stage3_ramp_epochs", 0.0))
        if ramp_epochs <= 0.0:
            return 1.0
        return self._epoch_ramp(start_epoch, start_epoch + ramp_epochs)

    def _stage3_lerp(self, base: float, target: float) -> float:
        return float(base) + (float(target) - float(base)) * self.final_stage3_progress()

    def thrust_gain_range(self) -> tuple[float, float]:
        """Rotor thrust-gain randomization band, widened by the stage-3 overlay."""
        base_min, base_max = (float(value) for value in self.cfg.final_thrust_gain_range)
        target_min, target_max = (
            float(value) for value in getattr(self.cfg, "final_stage3_thrust_gain_range", self.cfg.final_thrust_gain_range)
        )
        return self._stage3_lerp(base_min, target_min), self._stage3_lerp(base_max, target_max)

    def actuator_tau_range(self) -> tuple[float, float]:
        """Actuator lag (latency) band, widened by the stage-3 overlay."""
        base = (float(self.cfg.actuator_tau_min_s), float(self.cfg.actuator_tau_max_s))
        target_min, target_max = (
            float(value) for value in getattr(self.cfg, "final_stage3_actuator_tau_range", base)
        )
        return self._stage3_lerp(base[0], target_min), self._stage3_lerp(base[1], target_max)

    def thrust_loss_max_limit(self, base_loss_max: float) -> float:
        """Per-rotor thrust-loss ceiling, widened by the stage-3 overlay.

        Never tightens the stage value: the overlay target is only honoured where
        it exceeds what stage 3 already asks for.
        """
        target = float(getattr(self.cfg, "final_stage3_thrust_loss_max", base_loss_max))
        if target <= float(base_loss_max):
            return float(base_loss_max)
        return self._stage3_lerp(float(base_loss_max), target)

    def initial_error_multiplier(self) -> float:
        """Scale on reset position/attitude/velocity error envelopes."""
        return self._stage3_lerp(
            1.0, float(getattr(self.cfg, "final_stage3_initial_error_multiplier", 1.0))
        )

    def curriculum_stage_for_value(self, name: str | None = None) -> int:
        stage = int(self.cfg.curriculum_stage)
        if stage < 1:
            raise ValueError(f"curriculum_stage must be >= 1, got {self.cfg.curriculum_stage}")
        if not self.final_stage_overlay_active():
            return stage
        if name is not None and name.startswith("thrust_loss"):
            return max(1, int(getattr(self.cfg, "final_stage_thrust_loss_stage", 3)))
        return max(1, int(getattr(self.cfg, "final_stage_overlay_stage", 3)))

    def action_authority(self, start_stage: int, ramp_epochs: float) -> float:
        stage = self.curriculum_stage_for_value("action_authority")
        if stage < start_stage:
            return 0.0
        if stage > start_stage or ramp_epochs <= 0.0:
            return 1.0
        if self.final_stage_overlay_active() and int(self.cfg.curriculum_stage) < start_stage:
            elapsed = self._training_epoch() - float(self.cfg.final_stage_start_epoch)
        else:
            elapsed = self._training_epoch() - float(self.cfg.stage_ramp_start_epoch)
        return max(0.0, min(elapsed / float(ramp_epochs), 1.0))

    def new_dof_randomization_progress(self, start_stage: int, authority_ramp_epochs: float) -> float:
        stage = self.curriculum_stage_for_value("thrust_loss")
        if stage < start_stage:
            return 0.0
        if stage > start_stage:
            return 1.0
        if self.final_stage_overlay_active() and int(self.cfg.curriculum_stage) < start_stage:
            elapsed = self._training_epoch() - float(self.cfg.final_stage_start_epoch)
        else:
            elapsed = self._training_epoch() - float(self.cfg.stage_ramp_start_epoch)
        ramp_epochs = float(self.cfg.new_dof_randomization_ramp_epochs)
        if ramp_epochs <= 0.0:
            return 1.0
        return max(0.0, min((elapsed - authority_ramp_epochs) / ramp_epochs, 1.0))

    def _training_epoch(self) -> float:
        if self._training_epoch_cache_step == self._global_env_step:
            return self._training_epoch_cache_value
        local_epoch = float(self._global_env_step) / max(float(self.cfg.ramp_steps_per_epoch), 1.0)
        value = float(self.cfg.ramp_epoch_offset) + local_epoch
        self._training_epoch_cache_step = self._global_env_step
        self._training_epoch_cache_value = value
        return value

    def _epoch_ramp(self, start_epoch: float, end_epoch: float) -> float:
        if end_epoch <= start_epoch:
            return 1.0
        progress = (self._training_epoch() - float(start_epoch)) / (float(end_epoch) - float(start_epoch))
        return max(0.0, min(1.0, progress))

    def disturbance_weight(self) -> float:
        if self._disturbance_weight_cache_step == self._global_env_step:
            return self._disturbance_weight_cache_value
        initial_scale = float(
            self._stage_value(
                self.cfg.disturbance_ramp_initial_scale_by_stage,
                "disturbance_ramp_initial_scale_by_stage",
            )
        )
        start_epoch = self._stage_value(
            self.cfg.disturbance_ramp_start_epoch_by_stage,
            "disturbance_ramp_start_epoch_by_stage",
        )
        end_epoch = self._stage_value(
            self.cfg.disturbance_ramp_end_epoch_by_stage,
            "disturbance_ramp_end_epoch_by_stage",
        )
        progress = self._epoch_ramp(start_epoch, end_epoch)
        initial_scale = max(0.0, min(1.0, initial_scale))
        value = initial_scale + ((1.0 - initial_scale) * progress)
        self._disturbance_weight_cache_step = self._global_env_step
        self._disturbance_weight_cache_value = value
        return value


    def _make_task(self):
        task_name = self.cfg.task_name.lower()
        if task_name == "landing":
            return LandingTask(self, self.cfg.landing)
        if task_name == "takeoff":
            return TakeoffTask(self, self.cfg.takeoff)
        if task_name == "combined":
            return CombinedTask(self, self.cfg.combined)
        raise ValueError(
            f"Unsupported task_name '{self.cfg.task_name}'. Expected 'landing', 'takeoff', or 'combined'."
        )

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
        return self.get_cached_dones()

    def get_cached_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_dones_step != self._global_env_step or self._cached_dones is None:
            self._cached_dones = self.task.get_dones()
            self._cached_dones_step = self._global_env_step
        return self._cached_dones

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        elif len(env_ids) == 0:
            return

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self._is_first_sim_step:
            self._is_first_sim_step = False
        self.randomizer.reset(env_ids)
        self.task.reset_episode_state(env_ids)
        if hasattr(self, "_reset_observation_history_pending"):
            self._reset_observation_history_pending[env_ids] = True
            self._reset_observation_history_dirty = True
        self._cached_dones_step = -1

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "landing_ref_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.08, 0.08, 0.08)
                marker_cfg.prim_path = "/Visuals/Command/landing_reference_position"
                marker = marker_cfg.markers["cuboid"]
                if hasattr(marker, "visual_material"):
                    try:
                        marker.visual_material = sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.65, 1.0),
                            opacity=0.35,
                        )
                    except TypeError:
                        marker.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.65, 1.0))
                self.landing_ref_visualizer = VisualizationMarkers(marker_cfg)
            if self._debug_draw is None:
                try:
                    from omni.isaac.debug_draw import _debug_draw

                    self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                except Exception:
                    self._debug_draw = None
            self.goal_pos_visualizer.set_visibility(True)
            self.landing_ref_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "landing_ref_visualizer"):
                self.landing_ref_visualizer.set_visibility(False)
            if self._debug_draw is not None:
                self._debug_draw.clear_lines()

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.set_visibility(True)
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        if hasattr(self, "landing_ref_visualizer"):
            self.landing_ref_visualizer.visualize(self._trajectory_reference_pos_w())
        self._draw_force_debug_vectors()

    def _trajectory_reference_pos_w(self) -> torch.Tensor:
        if hasattr(self.task, "debug_reference_position_w"):
            return self.task.debug_reference_position_w()
        if hasattr(self.task, "reference_state"):
            ref_pos_w, _, _ = self.task.reference_state()
            return ref_pos_w
        return self._desired_pos_w

    def _draw_force_debug_vectors(self):
        if not self.cfg.thrust_vector_debug_vis or self._debug_draw is None:
            return
        if not hasattr(self, "_debug_thrust_w") or not self.vehicle.rotor_ids:
            return

        num_envs = min(int(self.cfg.thrust_vector_debug_num_envs), self.num_envs)
        if num_envs <= 0:
            self._debug_draw.clear_lines()
            return

        body_pos_w = self._robot.data.body_link_pos_w
        rotor_pos_w = body_pos_w[:num_envs, self.vehicle.rotor_ids, :]
        thrust_w = self._debug_thrust_w[:num_envs]
        moment_w = self._debug_moment_w[:num_envs]
        starts = []
        ends = []
        colors = []
        widths = []

        thrust_starts = rotor_pos_w.reshape(-1, 3)
        thrust_ends = (rotor_pos_w + self.cfg.thrust_vector_debug_scale * thrust_w).reshape(-1, 3)
        moment_along_thrust = torch.sum(moment_w * thrust_w, dim=-1).reshape(-1)
        thrust_colors = [
            self._moment_debug_color(moment_value) for moment_value in moment_along_thrust.detach().cpu().tolist()
        ]
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

        self._append_wheel_command_debug_vectors(body_pos_w, num_envs, starts, ends, colors, widths)

        self._debug_draw.clear_lines()
        if not starts:
            return
        self._debug_draw.draw_lines(starts, ends, colors, widths)

    def _append_wheel_command_debug_vectors(
        self,
        body_pos_w: torch.Tensor,
        num_envs: int,
        starts: list[list[float]],
        ends: list[list[float]],
        colors: list[tuple[float, float, float, float]],
        widths: list[float],
    ):
        wheel_action_name = "wheel_speed"
        if not self.cfg.wheel_command_debug_vis or wheel_action_name not in self.vehicle.action_schema.slices:
            return

        wheel_command = torch.nan_to_num(
            self.vehicle.action_term_values(wheel_action_name, filtered=True)[:num_envs],
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        if wheel_command.shape[1] < 2:
            return

        drive = wheel_command[:, 0:1].clamp(-1.0, 1.0)
        left_right = wheel_command[:, 1:2].clamp(-1.0, 1.0)
        base_pos_w = body_pos_w[:num_envs, self.vehicle.base_link, :]
        offset = torch.tensor(
            [0.0, 0.0, float(self.cfg.wheel_command_debug_z_offset)],
            device=self.device,
            dtype=base_pos_w.dtype,
        )
        command_origin_w = base_pos_w + offset

        forward_yaw_offset = float(self.vehicle.spec.forward_yaw_offset)
        forward_b = torch.tensor(
            [[cos(forward_yaw_offset), sin(forward_yaw_offset), 0.0]],
            device=self.device,
            dtype=base_pos_w.dtype,
        )
        left_b = torch.stack((-forward_b[:, 1], forward_b[:, 0], forward_b[:, 2]), dim=1)
        base_quat_w = self._robot.data.root_link_quat_w[:num_envs]
        forward_w = quat_rotate(base_quat_w, forward_b.expand(num_envs, -1))
        left_w = quat_rotate(base_quat_w, left_b.expand(num_envs, -1))

        scale = float(self.cfg.wheel_command_debug_scale)
        drive_starts = command_origin_w
        drive_ends = command_origin_w + scale * drive * forward_w
        drive_colors = [(0.1, 1.0, 0.25, 1.0)] * num_envs
        self._append_debug_arrows(drive_starts, drive_ends, drive_colors, starts, ends, colors, widths)

        lr_starts = command_origin_w + torch.tensor(
            [0.0, 0.0, 0.06],
            device=self.device,
            dtype=base_pos_w.dtype,
        )
        lr_ends = lr_starts + scale * left_right * left_w
        lr_colors = [(1.0, 0.85, 0.05, 1.0)] * num_envs
        self._append_debug_arrows(lr_starts, lr_ends, lr_colors, starts, ends, colors, widths)

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
