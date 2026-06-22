from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from numpy import pi

try:
    from .spaces import ActionSchema, ActionTerm, ObservationSchema, ObservationTerm
except ImportError:
    from spaces import ActionSchema, ActionTerm, ObservationSchema, ObservationTerm


@dataclass(frozen=True)
class RotorSpec:
    body_name: str
    action_name: str
    action_index: int
    spin_direction: float
    kT: float
    kM: float
    thrust_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    axis_body_name: str | None = None


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


@dataclass(frozen=True)
class ContactSpec:
    valid_body_names: tuple[str, ...]
    invalid_body_names: tuple[str, ...]


@dataclass(frozen=True)
class ObservationSourceSpec:
    name: str
    size: int
    source: str
    noise_scale: str | float = 0.0
    joint_group: str | None = None
    history: bool = True


@dataclass(frozen=True)
class AgreementMapSpec:
    name: str
    mapper: Callable | None = None
    weight_schedule: str | None = None


@dataclass(frozen=True)
class VehicleSpec:
    name: str
    base_body_name: str
    rotors: tuple[RotorSpec, ...]
    joint_groups: tuple[JointGroupSpec, ...]
    contacts: ContactSpec
    action_terms: tuple[ActionTerm, ...]
    observation_terms: tuple[ObservationSourceSpec, ...]
    agreement_maps: tuple[AgreementMapSpec, ...] = ()
    randomize_rotors_together: bool = True
    landing_tuck_joint_group: str | None = None
    landing_tuck_target: float = pi / 2
    thrust_loss_start_stage: int | None = 1
    takeoff_target_lin_vel_w: tuple[float, float, float] = (0.0, 0.0, 0.3)

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


ATMO_SPEC = VehicleSpec(
    name="atmo",
    base_body_name="base_link",
    rotors=(
        RotorSpec("rotor0", "rotor_thrust", 0, -1.0, kT=28.15, kM=0.018, axis_body_name="armr"),
        RotorSpec("rotor1", "rotor_thrust", 1, -1.0, kT=28.15, kM=0.018, axis_body_name="arml"),
        RotorSpec("rotor2", "rotor_thrust", 2, 1.0, kT=28.15, kM=0.018, axis_body_name="arml"),
        RotorSpec("rotor3", "rotor_thrust", 3, 1.0, kT=28.15, kM=0.018, axis_body_name="armr"),
    ),
    joint_groups=(
        JointGroupSpec(
            name="morph_tilt",
            joint_names=("base_to_arml", "base_to_armr"),
            action_name="morph_tilt",
            max_velocity=pi / 8,
            lower=0.0,
            upper=pi / 2,
            quantize_action=True,
            initial_position_range="initial_tilt_range",
            initial_velocity_range="initial_tilt_vel_range",
        ),
    ),
    contacts=ContactSpec(
        valid_body_names=("wheel0", "wheel1", "wheel2", "wheel3"),
        invalid_body_names=("base_link", "arml", "armr"),
    ),
    action_terms=(
        ActionTerm("rotor_thrust", 4, 0.0, 1.0),
        ActionTerm("morph_tilt", 1, 0.0, 1.0),
    ),
    observation_terms=(
        ObservationSourceSpec("root_pos", 3, "root_pos_local", "pos_noise_scale"),
        ObservationSourceSpec("target_pos", 3, "target_pos_local", "pos_noise_scale", history=False),
        ObservationSourceSpec("root_rotation_matrix", 9, "root_rotation_matrix", "rot_noise_scale"),
        ObservationSourceSpec("root_lin_vel_w", 3, "root_lin_vel_w", "lin_vel_noise_scale"),
        ObservationSourceSpec("root_ang_vel_b", 3, "root_ang_vel_b", "ang_vel_noise_scale"),
        ObservationSourceSpec("tilt_angle", 1, "joint_group_position", "tilt_noise_scale", "morph_tilt"),
    ),
    landing_tuck_joint_group="morph_tilt",
    thrust_loss_start_stage=2,
    takeoff_target_lin_vel_w=(0.0, 0.0, 0.3),
)
