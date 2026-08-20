from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any

from numpy import pi

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab_assets import ATMO_CFG

from .landing_task import LandingTaskCfg
from .combined_task import CombinedTaskCfg
from .m4_config import ROBOT, STAGE
from .takeoff_task import TakeoffTaskCfg
from .vehicle_specs import ATMO_SPEC, M4TII_SPEC, VehicleSpec

M4TII_USD_PATH = "/home/warren/Documents/m4tii/m4tii-isaac/m4tii-isaac.usd"
M4TII_USD_ROOT_PRIM = "/m4tii_urdf"


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


HIGH_TRACTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode=os.environ.get("M4_SURFACE_FRICTION_COMBINE_MODE", "multiply"),
    restitution_combine_mode="multiply",
    # Four fixed wheels must scrub laterally to yaw. With the 1 N*m wheel
    # effort limit, higher isotropic friction locked out skid-steer rotation
    # even though rolling straight remained easy. Keep the Gazebo world's
    # contact friction in step with these values.
    static_friction=_env_float("M4_SURFACE_STATIC_FRICTION", 0.3),
    dynamic_friction=_env_float("M4_SURFACE_DYNAMIC_FRICTION", 0.2),
    restitution=0.0,
)

GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode=os.environ.get("M4_SURFACE_FRICTION_COMBINE_MODE", "multiply"),
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
)


def _make_m4tii_spawn_usd() -> str:
    wrapper_path = Path(M4TII_USD_PATH).with_name("m4tii-isaac-wrapper.usda")
    root_prim_name = M4TII_USD_ROOT_PRIM.strip("/")
    wrapper_text = f"""#usda 1.0
(
    defaultPrim = "{root_prim_name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root_prim_name}" (
    prepend references = @{M4TII_USD_PATH}@<{M4TII_USD_ROOT_PRIM}>
)
{{
}}
"""
    if not wrapper_path.exists() or wrapper_path.read_text() != wrapper_text:
        wrapper_path.write_text(wrapper_text)
    return wrapper_path.as_posix()


M4TII_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_make_m4tii_spawn_usd(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.0,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            ".*hip_joint": pi / 2,
            ".*leg_joint": pi / 2,
            ".*wheel_joint": 0.0,
            ".*blade.*_joint": 0.0,
        },
    ),
    actuators={
        "morph_tilt": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_joint"],
            effort_limit=1.0e6,
            velocity_limit=pi / 8,
            stiffness=1.0e5,
            damping=1.0e3,
        ),
        "leg_dof": ImplicitActuatorCfg(
            joint_names_expr=[".*leg_joint"],
            effort_limit=150.0,
            velocity_limit=pi / 8,
            stiffness=80.0,
            damping=8.0,
        ),
        "wheel_spin": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            effort_limit=1.0,
            # Raised from 12.0 to cover the wheel spec's 18.0 rad/s
            # max_velocity now that the wheels track a velocity setpoint.
            velocity_limit=18.0,
            stiffness=0.0,
            # Velocity gain for the wheel_speed setpoint action: the implicit
            # PD's damping term is what tracks runtime.target_vel (stiffness
            # stays 0 -- no position control). 0.25 saturates the 1.0 N*m
            # effort limit at a 4 rad/s speed error. SIM-SIDE gain only,
            # checked for hardware-free sanity, tunable; the real Dynamixels
            # will run their own velocity-mode controller.
            damping=0.25,
        ),
        "free_spin": ImplicitActuatorCfg(
            joint_names_expr=[".*blade.*_joint"],
            effort_limit=1.0e-6,
            velocity_limit=1000.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


def _robot_cfg_with_extra_actuators(robot_cfg: Any, extra_actuators: dict[str, Any]) -> Any:
    actuators = dict(getattr(robot_cfg, "actuators", {}) or {})
    actuators.update(extra_actuators)
    return robot_cfg.replace(actuators=actuators)


def _robot_cfg_with_physics_material(robot_cfg: Any, material: Any) -> Any:
    spawn = getattr(robot_cfg, "spawn", None)
    if spawn is None or not hasattr(spawn, "replace"):
        return robot_cfg
    try:
        return robot_cfg.replace(spawn=spawn.replace(physics_material=material))
    except (TypeError, ValueError):
        return robot_cfg


def _robot_cfg_with_articulation_solver(
    robot_cfg: Any,
    *,
    position_iterations: int = 8,
    velocity_iterations: int = 4,
) -> Any:
    spawn = getattr(robot_cfg, "spawn", None)
    if spawn is None or not hasattr(spawn, "replace"):
        return robot_cfg
    articulation_props = getattr(spawn, "articulation_props", None)
    try:
        if articulation_props is not None and hasattr(articulation_props, "replace"):
            articulation_props = articulation_props.replace(
                solver_position_iteration_count=position_iterations,
                solver_velocity_iteration_count=velocity_iterations,
                sleep_threshold=0.0,
            )
        else:
            articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=position_iterations,
                solver_velocity_iteration_count=velocity_iterations,
                sleep_threshold=0.0,
                stabilization_threshold=0.001,
            )
        return robot_cfg.replace(spawn=spawn.replace(articulation_props=articulation_props))
    except (TypeError, ValueError):
        return robot_cfg


ATMO_WITH_WHEELS_CFG = _robot_cfg_with_extra_actuators(
    _robot_cfg_with_articulation_solver(_robot_cfg_with_physics_material(ATMO_CFG, HIGH_TRACTION_MATERIAL)),
    {
        "wheel_spin": ImplicitActuatorCfg(
            joint_names_expr=[".*_to_wheel[0-9]"],
            effort_limit=1.0,
            velocity_limit=18.0,
            stiffness=0.0,
            # MUST stay zero, unlike the m4tii velocity-setpoint wheels.
            #
            # ATMO's wheels are differential_duty: the adapter computes the
            # delivered torque itself, back-EMF included, and applies it as an
            # explicit effort. The implicit actuator's PD still runs on top of
            # whatever is applied, and its velocity target is zero, so any
            # damping here is a SECOND speed-dependent term the motor model
            # does not have -- at 0.25 it contributes -4.5 N*m at the 18 rad/s
            # no-load speed against a command whose whole range is +/-1.0 N*m,
            # which would swamp the wheels entirely. The back-EMF term in
            # _compute_duty_joint_targets is the only speed dependence there
            # should be.
            damping=0.0,
        ),
    },
)


@dataclass(frozen=True)
class M4VehicleProfile:
    """Vehicle-level defaults for one member of the M4 family."""

    name: str
    spec: VehicleSpec
    robot_cfg: Any
    robot_prim_path: str = "/World/envs/env_.*/Robot"

    curriculum_stage: int = 2
    reuse_last_stage_value: bool = True
    is_final_stage: bool | None = None
    final_stage_start_epoch: float | None = None
    final_stage_overlay_stage: int | None = None
    final_stage_thrust_loss_stage: int | None = None
    # Stage-3 hardening overlay. Complements the overlay above, which only ever
    # fires below stage 3.
    final_stage3_overlay_enabled: bool | None = None
    final_stage3_start_epoch: float | None = None
    # Per-vehicle, because it must exceed that vehicle's stage-3 thrust_loss_max
    # to widen anything at all -- the two profiles do not share a stage-3 value.
    final_stage3_thrust_loss_max: float | None = None

    # Dimensionless fractions of vehicle nominal thrust / rotor moment capacity.
    # Reduced from 0.05 / 0.005 for the stage-1 retrain: the disturbance
    # magnitude was overweighted relative to the new joint-dynamics and
    # estimation-noise randomizations, which now carry part of that burden.
    disturbance_force_scale: float = 0.035
    disturbance_moment_scale: float = 0.0035
    dist_force_cts_scale: float = 0.0
    dist_moment_cts_scale: float = 0.0
    thrust_loss_max: tuple[float, ...] = (0.0, 0.056, 0.12)
    initial_tilt_range: tuple[float, float] = (0.0, pi / 6)
    initial_leg_range: tuple[float, float] = (0.0, 0.0)
    initial_tilt_vel_range: tuple[float, float] = (0.0, 1.0)

    landing: Any = None
    takeoff: Any = None
    combined: Any = None

    def rl_games_name(self, task_name: str | None = None, curriculum_stage: int | None = None) -> str:
        parts = [self.name]
        if task_name:
            parts.append(task_name.lower())
        if curriculum_stage is not None:
            parts.append(f"stage{int(curriculum_stage)}")
        return "_".join(parts)

    def final_stage_enabled(self) -> bool:
        return False if self.is_final_stage is None else bool(self.is_final_stage)

    def final_stage_epoch(self) -> float:
        return 200.0 if self.final_stage_start_epoch is None else float(self.final_stage_start_epoch)

    def final_overlay_stage(self) -> int:
        return 3 if self.final_stage_overlay_stage is None else int(self.final_stage_overlay_stage)

    def final_thrust_loss_stage(self) -> int:
        return 3 if self.final_stage_thrust_loss_stage is None else int(self.final_stage_thrust_loss_stage)

    def final_stage3_enabled(self) -> bool:
        return False if self.final_stage3_overlay_enabled is None else bool(self.final_stage3_overlay_enabled)

    def final_stage3_epoch(self) -> float:
        return 400.0 if self.final_stage3_start_epoch is None else float(self.final_stage3_start_epoch)

    def final_stage3_thrust_loss(self) -> float:
        """Overlay thrust-loss ceiling.

        Defaults to a 1.4x widening of this profile's own stage-3 value rather
        than to a shared constant: a constant below the profile's stage-3 loss is
        silently a no-op, which is what a shared 0.20 would have been for m4tii.
        """
        if self.final_stage3_thrust_loss_max is not None:
            return float(self.final_stage3_thrust_loss_max)
        stage3_loss = float(self.thrust_loss_max[min(2, len(self.thrust_loss_max) - 1)])
        return stage3_loss * 1.4


def _landing_cfg_with_overrides(**overrides: Any) -> LandingTaskCfg:
    cfg = LandingTaskCfg()
    for name, value in overrides.items():
        if not hasattr(cfg, name):
            raise ValueError(f"LandingTaskCfg has no field '{name}'")
        setattr(cfg, name, value)
    return cfg


ATMO_PROFILE = M4VehicleProfile(
    name="atmo",
    spec=ATMO_SPEC,
    robot_cfg=ATMO_WITH_WHEELS_CFG,
    is_final_stage=True,
    final_stage_start_epoch=250.0,
    final_stage_overlay_stage=3,
    # Stage-3 only: final_stage3_overlay_active() requires curriculum_stage == 3
    # by construction, as the complement of final_stage_overlay_active(). ATMO
    # is training at stage 1, so this schedule is inert and moving it would
    # change a future stage-3 run on no evidence. Left alone deliberately.
    final_stage3_overlay_enabled=True,
    final_stage3_start_epoch=400.0,
    final_stage3_thrust_loss_max=0.20,
    dist_force_cts_scale=0.05,
    dist_moment_cts_scale=0.005,
    thrust_loss_max=(0.0, 0.056, 0.12),
    landing=_landing_cfg_with_overrides(
        post_landing_thrust_pen_scale=[-80.0, -80.0, -120.0],
        thrust_center_action_pen_scale=[0.0, 0.0, 0.0],
        thrust_center_loss_offset_pen_scale=[0.0, 0.0, 0.0],
    ),
    takeoff=TakeoffTaskCfg(),
    combined=CombinedTaskCfg(),
)


M4TII_PROFILE = M4VehicleProfile(
    name="m4tii",
    spec=M4TII_SPEC,
    robot_cfg=_robot_cfg_with_physics_material(M4TII_CFG, HIGH_TRACTION_MATERIAL),
    dist_force_cts_scale=0.05,
    dist_moment_cts_scale=0.005,
    # Stage 3 reduced 0.20 -> 0.12 (ATMO's value): at 0.20 the worst-case draw
    # compounded with the stage-3 shared thrust gain (0.75 low end) to leave
    # the weakest rotor at 0.60 of nominal against a ~0.486 hover -- episodes
    # with essentially no climb margin, and stage-3 training struggled.
    thrust_loss_max=(0.0, 0.056, 0.12),
    final_stage3_overlay_enabled=True,
    final_stage3_start_epoch=400.0,
    # Overlay target scaled down with the stage-3 base (was 0.28 over 0.20):
    # 0.17 keeps roughly the same 1.4x relative widening over 0.12.
    final_stage3_thrust_loss_max=0.17,
    initial_tilt_range=(0.0, pi / 6),
    initial_leg_range=(0.0, 0.0),
    landing=LandingTaskCfg(),
    takeoff=TakeoffTaskCfg(),
    combined=CombinedTaskCfg(),
)

VEHICLE_PROFILES = {
    ATMO_PROFILE.name: ATMO_PROFILE,
    M4TII_PROFILE.name: M4TII_PROFILE,
}


def _active_profile() -> M4VehicleProfile:
    try:
        profile = VEHICLE_PROFILES[ROBOT]
    except KeyError as exc:
        known = ", ".join(sorted(VEHICLE_PROFILES))
        raise ValueError(f"Unknown M4 robot '{ROBOT}'. Expected one of: {known}") from exc
    active_profile = replace(profile, curriculum_stage=STAGE)
    return active_profile


ACTIVE_PROFILE = _active_profile()


def apply_profile_to_rl_games_cfg(
    agent_cfg: dict,
    profile: M4VehicleProfile,
    *,
    task_name: str | None = None,
    curriculum_stage: int | None = None,
) -> dict:
    params = agent_cfg.setdefault("params", {})
    config = params.setdefault("config", {})
    config["name"] = profile.rl_games_name(
        task_name=task_name,
        curriculum_stage=curriculum_stage,
    )
    return agent_cfg
