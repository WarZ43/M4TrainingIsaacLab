from __future__ import annotations

import os

from dataclasses import dataclass

from numpy import pi

from .spaces import ActionSchema, ActionTerm, ObservationSchema, ObservationTerm


ATMO_KT = 28.15
ATMO_KM = 0.018
M4TII_KM = 0.018


@dataclass(frozen=True)
class RotorSpec:
    body_name: str
    action_index: int
    spin_direction: float
    kT: float
    kM: float
    thrust_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    axis_body_name: str | None = None
    # Optional per-rotor coefficients for the shared roll/pitch/yaw policy basis.
    control_mix: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class JointGroupSpec:
    name: str
    joint_names: tuple[str, ...]
    action_name: str
    max_velocity: float
    lower: float
    upper: float
    quantize_action: bool = False
    initial_position_range: str | tuple[float, float] | None = None
    initial_velocity_range: str | tuple[float, float] | None = None
    position_offset: float | tuple[float, ...] = 0.0
    position_scale: float | tuple[float, ...] = 1.0
    action_mapping: str = "direct"
    effort_limit: float | tuple[float, ...] = 0.0
    differential_drive_scale: float = 1.0
    differential_turn_scale: float = 1.0


@dataclass(frozen=True)
class ContactSpec:
    valid_body_names: tuple[str, ...]
    invalid_body_names: tuple[str, ...]


@dataclass(frozen=True)
class LegGeometrySpec:
    base_com: tuple[float, float, float]
    hip_xyz: tuple[tuple[float, float, float], ...]
    hip_rpy: tuple[tuple[float, float, float], ...]
    hip_axis_sign: tuple[float, ...]
    leg_xyz: tuple[tuple[float, float, float], ...]
    leg_rpy: tuple[tuple[float, float, float], ...]
    leg_axis_sign: tuple[float, ...]
    blade_xyz: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ObservationSourceSpec:
    name: str
    size: int
    source: str
    noise_scale: str | float = 0.0
    joint_group: str | None = None
    history: bool = True


@dataclass(frozen=True)
class VehicleSpec:
    name: str
    base_body_name: str
    rotors: tuple[RotorSpec, ...]
    joint_groups: tuple[JointGroupSpec, ...]
    contacts: ContactSpec
    action_terms: tuple[ActionTerm, ...]
    observation_terms: tuple[ObservationSourceSpec, ...]
    morph_joint_group: str | None = None
    leg_joint_group: str | None = None
    leg_geometry: LegGeometrySpec | None = None
    morph_target: float = pi / 2
    landing_joint_config: tuple[tuple[str, float | tuple[float, ...]], ...] = ()
    hover_joint_config: tuple[tuple[str, float | tuple[float, ...]], ...] = ()
    kT_hover_throttle: float | None = None
    rotor_mix_scale: tuple[float, float, float] = (0.25, 0.25, 0.20)
    forward_yaw_offset: float = 0.0
    ground_reference_height: float = 0.20

    def make_action_schema(self) -> ActionSchema:
        return ActionSchema(self.action_terms)

    def make_observation_schema(self, history: bool = True) -> ObservationSchema:
        return ObservationSchema(
            ObservationTerm(term.name, term.size) for term in self.observation_terms if term.history == history
        )

    @property
    def action_dim(self) -> int:
        return sum(term.size for term in self.action_terms)

    @property
    def observation_dim(self) -> int:
        return sum(term.size for term in self.observation_terms if term.history)

    @property
    def current_observation_dim(self) -> int:
        return sum(term.size for term in self.observation_terms if not term.history)


WHEEL_ACTION_TERMS = (
    # Differential wheel command: [drive, turn] in [-1, 1], mixed through the
    # differential wheel basis to a per-wheel normalized command.
    #
    # The SLOT name is deliberately stable across vehicles and contract
    # versions; what the number MEANS is carried by the group's
    # action_mapping, which is also how the codebase already distinguishes
    # torque from velocity:
    #
    #   differential_velocity -- normalized speed, scaled by max_velocity into
    #       a rad/s velocity target. Requires a transport that can close a
    #       velocity loop, i.e. one with encoders. This is what M4TII uses.
    #   differential_duty     -- normalized DUTY CYCLE, i.e. motor volts as a
    #       fraction of supply. This is what ATMO uses; see the ATMO wheel_dof
    #       group for the motor model and why.
    #
    # History: this was wheel_torque, then became a speed setpoint on the
    # assumption that the transport would be Dynamixels moving from current to
    # velocity mode. ATMO's transport is actually a RoboClaw driven open-loop
    # on duty cycle with NO ENCODER, so no velocity loop can be closed on
    # hardware at all and the speed-setpoint reading was never deliverable.
    ActionTerm("wheel_speed", 2, -1.0, 1.0),
)
COMMON_ROOT_OBSERVATIONS = (
    ObservationSourceSpec("root_pos", 3, "root_pos_local", "pos_noise_scale"),
    ObservationSourceSpec("root_rotation_matrix", 9, "root_rotation_matrix", "rot_noise_scale"),
    ObservationSourceSpec("root_lin_vel_w", 3, "root_lin_vel_w", "lin_vel_noise_scale"),
    ObservationSourceSpec("root_ang_vel_w", 3, "root_ang_vel_w", "ang_vel_noise_scale"),
)

COMMON_TASK_OBSERVATIONS = (
    ObservationSourceSpec("reference_accel", 3, "task_ref_accel_w", 0.0, history=False),
    ObservationSourceSpec("reference_pos_error", 3, "task_ref_pos_error_w", "pos_noise_scale", history=False),
    ObservationSourceSpec("reference_vel_error", 3, "task_vel_error_w", "lin_vel_noise_scale", history=False),
    ObservationSourceSpec("reference_yaw_accel", 1, "task_ref_yaw_accel", 0.0, history=False),
    ObservationSourceSpec("yaw_error", 1, "task_yaw_error", 0.0, history=False),
    ObservationSourceSpec("reference_yaw_rate_error", 1, "task_ref_yaw_rate_error", 0.0, history=False),
)

COMMON_VEHICLE_OBSERVATIONS = (
    ObservationSourceSpec(
        "thrust_wrench_forward_matrix", 24, "thrust_wrench_forward_matrix_w", 0.0, history=False
    ),
    ObservationSourceSpec(
        "thrust_wrench_allocation_matrix", 24, "thrust_wrench_allocation_matrix_w", 0.0, history=False
    ),
    ObservationSourceSpec("thrust_center_xy", 2, "thrust_center_xy_b", 0.0, history=False),
)

ATMO_SPEC = VehicleSpec(
    name="atmo",
    base_body_name="base_link",
    rotors=(
        # control_mix = (roll, pitch, yaw). Roll and yaw signs FLIPPED 2026-08-19
        # (pitch unchanged) -- was (1,1,-1) (-1,-1,-1) (-1,1,1) (1,-1,1).
        RotorSpec("rotor0", 0, -1.0, kT=ATMO_KT, kM=ATMO_KM, control_mix=(-1.0, 1.0, 1.0)),
        RotorSpec("rotor1", 1, -1.0, kT=ATMO_KT, kM=ATMO_KM, control_mix=(1.0, -1.0, 1.0)),
        RotorSpec("rotor2", 2, 1.0, kT=ATMO_KT, kM=ATMO_KM, control_mix=(1.0, 1.0, -1.0)),
        RotorSpec("rotor3", 3, 1.0, kT=ATMO_KT, kM=ATMO_KM, control_mix=(-1.0, -1.0, -1.0)),
    ),
    joint_groups=(
        JointGroupSpec(
            name="morph_tilt",
            joint_names=("base_to_arml", "base_to_armr"),
            action_name="tilt_mean",
            max_velocity=pi / 8,  # 22.5 deg/s -- matches a healthy motor at full
            # tilt (measured 2026-08-19). Acceleration is not modeled yet.
            lower=0.0,
            upper=pi / 2,
            quantize_action=True,
            initial_position_range="initial_tilt_range",
            initial_velocity_range="initial_tilt_vel_range",
        ),
        JointGroupSpec(
            name="wheel_dof",
            # The differential wheel basis expects front-left, front-right,
            # rear-left, rear-right. In the ATMO URDF, left wheels are on arml
            # and right wheels are on armr. ATMO's rolling-forward side is
            # body -x, so wheel1/wheel3 are the front pair. The arms tuck
            # about mirrored x-axes, so the right-side wheel joints need
            # opposite sim signs for one virtual drive command to roll each
            # pair together.
            joint_names=("arml_to_wheel1", "armr_to_wheel3", "arml_to_wheel2", "armr_to_wheel0"),
            action_name="wheel_speed",
            # No-load speed at full duty. Same number as the old velocity cap,
            # and it keeps that meaning: it is the speed at which back-EMF
            # cancels the applied volts and the motor stops producing torque.
            max_velocity=18.0,
            lower=-1.0e9,
            upper=1.0e9,
            position_scale=(1.0, -1.0, 1.0, -1.0),
            # RoboClaw, open-loop duty cycle, NO ENCODER. The command is
            # effectively motor volts, so the wheel is a voltage source rather
            # than either a speed source or a torque source:
            #
            #     tau = kt * (V - ke*w) / R
            #         = tau_stall * (duty - w / w_no_load)
            #
            # A velocity setpoint cannot be delivered by this transport at all
            # -- RoboClaw's velocity PID needs quadrature feedback the vehicle
            # does not have -- and plain torque would over-deliver at speed by
            # omitting the back-EMF term. differential_duty applies the line
            # above, so torque falls off linearly with wheel speed and reaches
            # zero at max_velocity, exactly as the real motor does.
            action_mapping="differential_duty",
            # Torque at full duty from rest, and also the current-limit clamp
            # (they are collapsed into one number deliberately: the RoboClaw
            # limits current, and separating stall torque from the limit would
            # mean inventing a distinction neither value has been measured to
            # support -- if motor data later says otherwise, split them). 1.0
            # N*m is the wheel torque scale this project already runs: it is
            # the actuator effort_limit in vehicle_profiles.py and the value
            # HIGH_TRACTION_MATERIAL's friction coefficients were tuned
            # against, so reusing it leaves that tuning valid.
            effort_limit=1.0,
            differential_drive_scale=1.0,
            # Halved for the same reason as the M4TII wheel group: skid-steer
            # yaw torque is friction-saturated far below the speed cap.
            differential_turn_scale=0.5,
        ),
    ),
    contacts=ContactSpec(
        valid_body_names=("wheel0", "wheel1", "wheel2", "wheel3"),
        invalid_body_names=("base_link", "arml", "armr"),
    ),
    action_terms=(
        ActionTerm("lift", 1, -1.0, 1.0),
        ActionTerm("roll_pitch_yaw", 3, -1.0, 1.0),
        ActionTerm("tilt_mean", 1, -1.0, 1.0),
        *WHEEL_ACTION_TERMS,
    ),
    observation_terms=(
        *COMMON_ROOT_OBSERVATIONS,
        ObservationSourceSpec(
            "tilt_angle",
            1,
            "joint_group_position",
            "tilt_noise_scale",
            "morph_tilt",
        ),
        *COMMON_TASK_OBSERVATIONS,
        *COMMON_VEHICLE_OBSERVATIONS,
        ObservationSourceSpec("morph_trig", 2, "morph_trig", 0.0, history=False),
    ),
    morph_joint_group="morph_tilt",
    landing_joint_config=(("morph_tilt", pi / 2),),
    hover_joint_config=(("morph_tilt", 0.0),),
    # Physical PX4/Gazebo rotor-order basis.
    #
    # 0.5 restores exact reachability: the basis can reproduce ANY raw
    # normalized 4-rotor command from bounded deltas, which is why it was
    # chosen originally.
    #
    # The 0.3 experiment is FALSIFIED, twice over. The hypothesis was that
    # clipping throttles learning -- at 0.5, 91.5% of broadly-sampled actions
    # clip at least one rotor channel against 47.7% at M4TII's 0.25, and a
    # clipped channel has zero gradient. The measurement was real; the
    # conclusion was not. Two runs differing only in this value crossed the 5%
    # takeoff threshold at epoch 100 (scale 0.5, run 07-03-29) and epoch 108
    # (scale 0.3, run 12-59-39). No gain, slightly later. Clipping is not the
    # bottleneck.
    #
    # It also caused a live defect: this was changed here and never propagated,
    # so training ran at 0.3 while contracts/atmo_combined_v1.json -- which
    # sim/atmo_sim/params.py reads -- stayed at 0.5. Every offline evaluation
    # in between was mis-scaling attitude by 1.67x, silently, with plausible
    # trajectories throughout.
    #
    # CONTRACT: this value is exported in contracts/atmo_combined_v1.json and
    # read by sim/atmo_sim/params.py. Changing it here without re-exporting is
    # exactly the failure described above. Re-export, do not hand-edit.
    rotor_mix_scale=(0.5, 0.5, 0.5),
    # ATMO faces 180 deg from its direction of travel because of this pi:
    # the yaw reference is `heading - forward_yaw_offset`, so the drive
    # target ends up BEHIND the vehicle. Overridable to test whether the
    # offset is a real chassis convention or a defect.
    forward_yaw_offset=float(os.environ.get("M4_ATMO_YAW_OFFSET", pi)),
)

M4TII_LEG_GEOMETRY = LegGeometrySpec(
    base_com=(-0.0086577, 0.00023257, -0.013926),
    hip_xyz=(
        (0.1615, 0.09457, 0.03975),
        (0.1615, -0.09451, 0.03975),
        (-0.168, 0.09457, 0.03975),
        (-0.168, -0.09457, 0.03975),
    ),
    hip_rpy=(
        (1.5708, 0.0, 1.5708),
        (1.5708, 0.0, -1.5708),
        (1.5708, 0.0, 1.5708),
        (1.5708, 0.0, -1.5708),
    ),
    hip_axis_sign=(1.0, 1.0, 1.0, 1.0),
    leg_xyz=(
        (0.02428, 0.0, 0.03943),
        (0.02428, 0.0, -0.03943),
        (0.01975, 0.0, -0.03493),
        (0.02325, 0.0, 0.03493),
    ),
    leg_rpy=(
        (3.1415, -1.5708, 0.0),
        (-3.1415, -1.5708, 0.0),
        (3.1415, -1.5708, 0.0),
        (-3.1415, -1.5708, 0.0),
    ),
    leg_axis_sign=(-1.0, 1.0, 1.0, -1.0),
    blade_xyz=(
        (0.0, 0.13385, 0.055259),
        (0.0, 0.13385, 0.054159),
        (0.0, 0.13385, 0.060895),
        (0.0, 0.13385, 0.058329),
    ),
)


M4TII_SPEC = VehicleSpec(
    name="m4tii",
    base_body_name="base_link",
    rotors=(
        RotorSpec(
            "front_left_blade_link",
            0,
            1.0,
            kT=0.0,
            kM=M4TII_KM,
            axis_body_name="front_left_wheel_link",
            control_mix=(-1.0, 1.0, 1.0),
        ),
        RotorSpec(
            "front_right_blade_link",
            1,
            -1.0,
            kT=0.0,
            kM=M4TII_KM,
            thrust_axis=(0.0, 0.0, -1.0),
            axis_body_name="front_right_wheel_link",
            control_mix=(1.0, 1.0, -1.0),
        ),
        RotorSpec(
            "rear_left_bladel_link",
            2,
            -1.0,
            kT=0.0,
            kM=M4TII_KM,
            axis_body_name="rear_left_wheel_link",
            control_mix=(-1.0, -1.0, -1.0),
        ),
        RotorSpec(
            "rear_right_blade_link",
            3,
            1.0,
            kT=0.0,
            kM=M4TII_KM,
            thrust_axis=(0.0, 0.0, -1.0),
            axis_body_name="rear_right_wheel_link",
            control_mix=(1.0, -1.0, 1.0),
        ),
    ),
    joint_groups=(
        JointGroupSpec(
            name="morph_tilt",
            joint_names=(
                "front_left_hip_joint",
                "front_right_hip_joint",
                "rear_left_hip_joint",
                "rear_right_hip_joint",
            ),
            action_name="tilt_mean",
            max_velocity=pi / 8,
            lower=0.0,
            upper=pi / 2,
            quantize_action=True,
            initial_position_range="initial_tilt_range",
            initial_velocity_range="initial_tilt_vel_range",
            position_offset=pi / 2,
            position_scale=-1.0,
            action_mapping="morph_tilt_basis",
        ),
        JointGroupSpec(
            name="leg_dof",
            joint_names=(
                "front_left_leg_joint",
                "front_right_leg_joint",
                "rear_left_leg_joint",
                "rear_right_leg_joint",
            ),
            action_name="thrust_center_xy",
            max_velocity=pi / 8,
            lower=0.0,
            upper=pi / 2,
            initial_position_range="initial_leg_range",
            action_mapping="thrust_center_shift",
        ),
        JointGroupSpec(
            name="wheel_dof",
            joint_names=(
                "front_left_wheel_joint",
                "front_right_wheel_joint",
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
            ),
            action_name="wheel_speed",
            max_velocity=18.0,
            lower=-1.0e9,
            upper=1.0e9,
            action_mapping="differential_velocity",
            # Skid-steer yaw saturates on friction well below the wheel-speed
            # cap: past breakaway the wheels just slide and extra differential
            # speed adds nothing, which made full-range turn commands pure
            # bang-bang at +-18 rad/s. Halving the turn authority keeps the
            # usable part of the curve across the action range and halves the
            # slam amplitude on hardware.
            differential_turn_scale=0.5,
        ),
    ),
    contacts=ContactSpec(
        valid_body_names=(
            "front_left_wheel_link",
            "front_right_wheel_link",
            "rear_left_wheel_link",
            "rear_right_wheel_link",
        ),
        invalid_body_names=(
            "base_link",
            "front_left_hip_link",
            "front_left_leg_link",
            "front_left_blade_link",
            "front_right_hip_link",
            "front_right_leg_link",
            "front_right_blade_link",
            "rear_left_hip_link",
            "rear_left_leg_link",
            "rear_left_bladel_link",
            "rear_right_hip_link",
            "rear_right_leg_link",
            "rear_right_blade_link",
        ),
    ),
    action_terms=(
        ActionTerm("lift", 1, -1.0, 1.0),
        ActionTerm("roll_pitch_yaw", 3, -1.0, 1.0),
        ActionTerm("tilt_mean", 1, -1.0, 1.0),
        ActionTerm("tilt_balance", 3, -1.0, 1.0),
        ActionTerm("thrust_center_xy", 2, -1.0, 1.0),
        *WHEEL_ACTION_TERMS,
    ),
    observation_terms=(
        *COMMON_ROOT_OBSERVATIONS,
        ObservationSourceSpec(
            "tilt_angle",
            4,
            "joint_group_position",
            "tilt_noise_scale",
            "morph_tilt",
        ),
        ObservationSourceSpec(
            "leg_dof_angle",
            4,
            "joint_group_position",
            "tilt_noise_scale",
            "leg_dof",
        ),
        *COMMON_TASK_OBSERVATIONS,
        *COMMON_VEHICLE_OBSERVATIONS,
        ObservationSourceSpec("morph_trig", 16, "morph_trig", 0.0, history=False),
    ),
    morph_joint_group="morph_tilt",
    leg_joint_group="leg_dof",
    leg_geometry=M4TII_LEG_GEOMETRY,
    morph_target=pi / 2,
    landing_joint_config=(
        ("morph_tilt", pi / 2),
        ("leg_dof", 0.0),
    ),
    hover_joint_config=(
        ("morph_tilt", 0.0),
        ("leg_dof", 0.0),
    ),
    kT_hover_throttle=0.5,
    ground_reference_height=0.25,
)
