from __future__ import annotations

import torch


class M4Randomizer:
    """Common disturbance and actuator randomization for M4 tasks."""

    def __init__(self, env):
        self.env = env
        self.task = None
        self.task_config = None
        self.task_force_scale = None
        self.task_moment_scale = None
        self.task_cts_force_scale = None
        self.task_cts_moment_scale = None
        self.reset_task = None
        (
            self._disturbance_batch_probabilities,
            self._disturbance_batch_scale_multipliers,
            self._thrust_loss_batch_scale_multipliers,
        ) = (
            self._make_disturbance_batch_config()
        )
        env._disturbance_force = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._disturbance_moment = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._disturbance_force_cts = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._disturbance_moment_cts = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._disturbance_batch_scale = torch.ones(env.num_envs, 1, 1, device=env.device)
        # Nominal COMs, captured once so each reset offsets from the model
        # value instead of compounding on the previous episode's draw.
        self._nominal_coms = None
        self._base_body_index = None
        env._thrust_loss_batch_scale = torch.ones(env.num_envs, 1, device=env.device)
        env._push_time = torch.zeros(env.num_envs, device=env.device)
        env._push_duration = torch.zeros(env.num_envs, device=env.device)
        env._push_end_time = torch.zeros(env.num_envs, device=env.device)
        env._push_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._wheel_shape_indices = None
        self._nominal_wheel_materials = None
        self._drive_material_buckets = None

    def bind_task(self, task) -> None:
        self.task = task
        self.task_config = task.cfg
        self.task_force_scale = getattr(task, "disturbance_force_scale", None)
        self.task_moment_scale = getattr(task, "disturbance_moment_scale", None)
        self.task_cts_force_scale = getattr(task, "disturbance_cts_force_scale", None)
        self.task_cts_moment_scale = getattr(task, "disturbance_cts_moment_scale", None)
        self.reset_task = task.reset_initial_state

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self._refresh_disturbance_batch_config()
        if self.env.cfg.randomize:
            self._reset_randomized(env_ids)
        else:
            self._reset_deterministic(env_ids)

    def _refresh_disturbance_batch_config(self) -> None:
        (
            self._disturbance_batch_probabilities,
            self._disturbance_batch_scale_multipliers,
            self._thrust_loss_batch_scale_multipliers,
        ) = self._make_disturbance_batch_config()

    def _disturbance_scales(self) -> tuple[float, float]:
        env = self.env
        force_scale = env.cfg.disturbance_force_scale
        moment_scale = env.cfg.disturbance_moment_scale
        if self.task_force_scale is not None:
            force_scale = self.task_force_scale()
        if self.task_moment_scale is not None:
            moment_scale = self.task_moment_scale()
        return force_scale, moment_scale

    def _continuous_disturbance_scales(self) -> tuple[float, float]:
        env = self.env
        force_scale = env.cfg.dist_force_cts_scale
        moment_scale = env.cfg.dist_moment_cts_scale
        if self.task_cts_force_scale is not None:
            force_scale = self.task_cts_force_scale()
        if self.task_cts_moment_scale is not None:
            moment_scale = self.task_cts_moment_scale()
        return force_scale, moment_scale

    def _sample_unit_vectors(self, shape: tuple[int, int, int]) -> torch.Tensor:
        direction = torch.normal(0.0, 1.0, size=shape, device=self.env.device)
        return direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-6)

    def _make_disturbance_batch_config(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env = self.env
        task_cfg = self.task_config
        probabilities_source = getattr(
            task_cfg,
            "disturbance_batch_probabilities",
            getattr(env.cfg, "disturbance_batch_probabilities", (1.0,)),
        )
        scale_multipliers_source = getattr(
            task_cfg,
            "disturbance_batch_scale_multipliers",
            getattr(env.cfg, "disturbance_batch_scale_multipliers", (1.0,)),
        )
        thrust_loss_scale_source = getattr(
            task_cfg,
            "thrust_loss_batch_scale_multipliers",
            getattr(env.cfg, "thrust_loss_batch_scale_multipliers", scale_multipliers_source),
        )
        probabilities_cfg = self._stage_config_value(
            probabilities_source,
            "disturbance_batch_probabilities",
        )
        scale_multipliers_cfg = self._stage_config_value(
            scale_multipliers_source,
            "disturbance_batch_scale_multipliers",
        )
        thrust_loss_scale_multipliers_cfg = self._stage_config_value(
            thrust_loss_scale_source,
            "thrust_loss_batch_scale_multipliers",
        )
        probabilities = tuple(float(value) for value in probabilities_cfg)
        scale_multipliers = tuple(float(value) for value in scale_multipliers_cfg)
        thrust_loss_scale_multipliers = tuple(float(value) for value in thrust_loss_scale_multipliers_cfg)
        if len(probabilities) == 0:
            raise ValueError("disturbance_batch_probabilities must contain at least one bucket")
        if len(probabilities) != len(scale_multipliers):
            raise ValueError(
                "disturbance_batch_probabilities and disturbance_batch_scale_multipliers must have the same length"
            )
        if len(probabilities) != len(thrust_loss_scale_multipliers):
            raise ValueError(
                "disturbance_batch_probabilities and thrust_loss_batch_scale_multipliers must have the same length"
            )
        if any(value < 0.0 for value in probabilities):
            raise ValueError("disturbance_batch_probabilities must be non-negative")
        if any(value < 0.0 for value in scale_multipliers):
            raise ValueError("disturbance_batch_scale_multipliers must be non-negative")
        if any(value < 0.0 for value in thrust_loss_scale_multipliers):
            raise ValueError("thrust_loss_batch_scale_multipliers must be non-negative")
        probability_sum = sum(probabilities)
        if probability_sum <= 0.0:
            raise ValueError("disturbance_batch_probabilities must sum to a positive value")

        normalized_probabilities = [value / probability_sum for value in probabilities]
        return (
            torch.tensor(normalized_probabilities, dtype=torch.float, device=env.device),
            torch.tensor(scale_multipliers, dtype=torch.float, device=env.device),
            torch.tensor(thrust_loss_scale_multipliers, dtype=torch.float, device=env.device),
        )

    def _stage_config_value(self, values, name: str):
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            return values
        if not isinstance(values[0], (list, tuple)):
            return values

        if hasattr(self.env, "curriculum_stage_for_value"):
            stage = int(self.env.curriculum_stage_for_value(name))
        else:
            stage = int(getattr(self.env.cfg, "curriculum_stage", 1))
        if stage < 1:
            raise ValueError(f"curriculum_stage must be >= 1, got {stage}")
        stage_idx = stage - 1
        try:
            return values[stage_idx]
        except IndexError as exc:
            if getattr(self.env.cfg, "reuse_last_stage_value", True):
                return values[-1]
            raise ValueError(f"curriculum_stage={stage} has no entry for {name}") from exc

    def _sample_disturbance_batch_scale(self, num_resets: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._disturbance_batch_scale_multipliers.numel() == 1:
            return (
                self._disturbance_batch_scale_multipliers.reshape(1, 1, 1).repeat(num_resets, 1, 1),
                self._thrust_loss_batch_scale_multipliers.reshape(1, 1).repeat(num_resets, 1),
            )
        bucket_ids = torch.multinomial(
            self._disturbance_batch_probabilities,
            num_resets,
            replacement=True,
        )
        return (
            self._disturbance_batch_scale_multipliers[bucket_ids].reshape(num_resets, 1, 1),
            self._thrust_loss_batch_scale_multipliers[bucket_ids].reshape(num_resets, 1),
        )

    def _reset_actuators(self, env_ids: torch.Tensor, randomized: bool) -> None:
        env = self.env
        vehicle = env.vehicle
        nominal_kT = vehicle._nominal_kT_tensor
        nominal_kM = vehicle._nominal_kM_tensor
        rotor_count = vehicle.rotor_count

        thrust_scale = torch.ones(len(env_ids), rotor_count, device=env.device)
        if randomized and env.curriculum_stage_for_value("thrust_loss") >= 3:
            gain_min, gain_max = env.thrust_gain_range()
            if gain_min <= 0.0 or gain_max < gain_min:
                raise ValueError(
                    f"final_thrust_gain_range must satisfy 0 < min <= max, got {(gain_min, gain_max)}"
                )
            thrust_scale *= torch.empty(len(env_ids), 1, device=env.device).uniform_(gain_min, gain_max)

        env.kM[env_ids] = nominal_kM

        if randomized and env.curriculum_stage_for_value("thrust_loss") >= env.cfg.thrust_loss_start_stage:
            loss_max = float(env._stage_value(env.cfg.thrust_loss_max, "thrust_loss_max"))
            stage = env.curriculum_stage_for_value("thrust_loss")
            authority_ramp_epochs = None
            if stage == env.cfg.morph_bias_start_stage:
                authority_ramp_epochs = env.cfg.morph_balance_authority_ramp_epochs
            elif stage == env.cfg.thrust_center_start_stage:
                authority_ramp_epochs = env.cfg.thrust_center_authority_ramp_epochs
            if authority_ramp_epochs is not None and isinstance(env.cfg.thrust_loss_max, (list, tuple)):
                previous_index = max(min(stage - 2, len(env.cfg.thrust_loss_max) - 1), 0)
                previous_loss_max = float(env.cfg.thrust_loss_max[previous_index])
                progress = env.new_dof_randomization_progress(stage, authority_ramp_epochs)
                loss_max = previous_loss_max + (loss_max - previous_loss_max) * progress
            # Widened by the stage-3 overlay, after the intra-stage authority
            # ramp above so the two do not compound into a double ramp.
            loss_max = env.thrust_loss_max_limit(loss_max)
            if loss_max < 0.0 or loss_max > 1.0:
                raise ValueError(f"thrust_loss_max must be in [0, 1], got {loss_max}")
            if loss_max > 0.0:
                loss_max_by_env = (loss_max * env._thrust_loss_batch_scale[env_ids]).clamp(max=1.0)
                thrust_scale *= 1.0 - torch.rand(len(env_ids), rotor_count, device=env.device) * loss_max_by_env

        minimum_thrust_scale = float(env.cfg.minimum_thrust_scale)
        if minimum_thrust_scale <= 0.0 or minimum_thrust_scale > 1.0:
            raise ValueError(f"minimum_thrust_scale must be in (0, 1], got {minimum_thrust_scale}")
        env.kT[env_ids] = nominal_kT * thrust_scale.clamp_min(minimum_thrust_scale)

        for runtime in vehicle.joint_groups.values():
            runtime.max_velocity[env_ids] = runtime.spec.max_velocity
            if runtime.zero_offset is not None:
                runtime.zero_offset[env_ids] = 0.0
        if randomized and bool(getattr(env.cfg, "randomize_joint_dynamics", False)):
            self._randomize_joint_dynamics(env_ids)

    def _reset_drive_friction(self, env_ids: torch.Tensor, randomized: bool) -> None:
        env = self.env
        if not hasattr(env, "_combined_mode") or not env.vehicle.spec.contacts.valid_body_names:
            return

        if self._wheel_shape_indices is None:
            left_shapes = []
            right_shapes = []
            shape_offset = 0
            wheel_names = set(env.vehicle.spec.contacts.valid_body_names)
            if env.vehicle.spec.name == "atmo":
                # ATMO's URDF numbers wheels by actuator order rather than
                # encoding the side in the body name: wheel1/wheel2 are left,
                # while wheel0/wheel3 are right.
                left_wheel_names = {"wheel1", "wheel2"}
                right_wheel_names = {"wheel0", "wheel3"}
            else:
                left_wheel_names = {name for name in wheel_names if "left" in name}
                right_wheel_names = {name for name in wheel_names if "right" in name}
            for link_path in env._robot.root_physx_view.link_paths[0]:
                shape_count = env._robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes
                body_name = link_path.rsplit("/", 1)[-1]
                if body_name in wheel_names:
                    if body_name in left_wheel_names:
                        target = left_shapes
                    elif body_name in right_wheel_names:
                        target = right_shapes
                    else:
                        target = None
                    if target is not None:
                        target.extend(range(shape_offset, shape_offset + shape_count))
                shape_offset += shape_count
            if not left_shapes or not right_shapes:
                raise RuntimeError("Could not resolve left and right wheel collision shapes for friction randomization")
            self._wheel_shape_indices = (
                torch.tensor(left_shapes, dtype=torch.long),
                torch.tensor(right_shapes, dtype=torch.long),
            )

        if self._drive_material_buckets is None:
            count = 64
            dynamic_min, dynamic_max = self.task_config.drive_dynamic_friction_range
            static_min, static_max = self.task_config.drive_static_friction_range
            spread = float(self.task_config.drive_side_friction_spread)
            base_dynamic = torch.empty(count, 1).uniform_(dynamic_min, dynamic_max)
            base_static = torch.empty(count, 1).uniform_(static_min, static_max)
            side_scale = torch.empty(count, 2).uniform_(1.0 - spread, 1.0 + spread)
            dynamic = (base_dynamic * side_scale).clamp(dynamic_min, dynamic_max)
            static = torch.maximum(
                (base_static * side_scale).clamp(static_min, static_max),
                dynamic,
            )
            restitution = torch.zeros_like(dynamic)
            self._drive_material_buckets = torch.stack((static, dynamic, restitution), dim=-1)

        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        materials = env._robot.root_physx_view.get_material_properties()
        left_shapes, right_shapes = self._wheel_shape_indices
        all_wheel_shapes = torch.cat((left_shapes, right_shapes)).to(materials.device)
        if self._nominal_wheel_materials is None:
            self._nominal_wheel_materials = materials[0, all_wheel_shapes].clone()
        material_rows = env_ids_cpu.to(materials.device)
        materials[material_rows[:, None], all_wheel_shapes[None, :]] = self._nominal_wheel_materials

        if randomized:
            drive_mask = env._combined_mode[env_ids] == self.task.DRIVE
            drive_ids = env_ids_cpu[drive_mask.cpu()]
            if len(drive_ids) > 0:
                bucket_ids = torch.randint(0, len(self._drive_material_buckets), (len(drive_ids),))
                samples = self._drive_material_buckets[bucket_ids].to(materials.device)
                drive_rows = drive_ids.to(materials.device)
                for side, shape_ids in enumerate((left_shapes, right_shapes)):
                    shapes = shape_ids.to(materials.device)
                    materials[drive_rows[:, None], shapes[None, :]] = samples[:, side, None, :]

        env._robot.root_physx_view.set_material_properties(
            materials,
            env_ids_cpu,
        )

    def _joint_group_spread_authority(self, runtime) -> float:
        """Authority over the action that can correct spread WITHIN this group.

        Asymmetry between the joints of a group is only correctable through that
        group's balance channel, and both channels are held at zero until their
        start stage. Returning the authority here lets every per-joint draw ramp
        in exactly as the actuator that answers it unlocks. Groups with no
        balance channel get 0.0, i.e. shared draws forever.
        """
        env = self.env
        if runtime.is_morph_basis:
            return float(
                env.action_authority(
                    env.cfg.morph_bias_start_stage,
                    env.cfg.morph_balance_authority_ramp_epochs,
                )
            )
        if runtime.is_thrust_center:
            return float(
                env.action_authority(
                    env.cfg.thrust_center_start_stage,
                    env.cfg.thrust_center_authority_ramp_epochs,
                )
            )
        return 0.0

    def _sample_shared_plus_spread(
        self,
        env_ids: torch.Tensor,
        like: torch.Tensor,
        low: float,
        high: float,
        authority: float,
    ) -> torch.Tensor:
        """One shared draw per env, plus a per-joint deviation scaled by authority.

        At authority 0 every joint in the group gets the SAME value, drawn from
        the full range -- stage 1 still sees slow servos and large offsets, it
        just sees them equally on all four corners. At authority 1 the joints are
        fully independent. The shared component is never scaled, so widening the
        curriculum never weakens the common-mode randomization.
        """
        target = like[env_ids]
        shared = torch.empty(target.shape[0], 1, device=target.device, dtype=target.dtype)
        shared.uniform_(low, high)
        if authority <= 0.0:
            return shared.expand_as(target).clone()
        per_joint = torch.empty_like(target).uniform_(low, high)
        return shared + (per_joint - shared) * authority

    def _randomize_joint_dynamics(self, env_ids: torch.Tensor) -> None:
        env = self.env
        ceiling_low, ceiling_high = (float(value) for value in env.cfg.joint_velocity_ceiling_range)
        if ceiling_low <= 0.0 or ceiling_high < ceiling_low:
            raise ValueError(
                f"joint_velocity_ceiling_range must satisfy 0 < min <= max, got {(ceiling_low, ceiling_high)}"
            )
        hip_offset_range = tuple(float(value) for value in env.cfg.hip_zero_offset_range)
        leg_offset_range = tuple(float(value) for value in env.cfg.leg_zero_offset_range)
        for runtime in env.vehicle.joint_groups.values():
            # The measured servo numbers describe the position-controlled hips
            # and legs only. Wheels spin freely far above this ceiling whether
            # torque-driven or velocity-setpoint (18 rad/s spec vs the 0.30-0.45
            # servo band); clamping them here silently stalled all driving when
            # the wheels moved off the torque mapping.
            if runtime.is_torque or runtime.is_velocity:
                continue
            authority = self._joint_group_spread_authority(runtime)
            runtime.max_velocity[env_ids] = self._sample_shared_plus_spread(
                env_ids, runtime.max_velocity, ceiling_low, ceiling_high, authority
            )
            if runtime.zero_offset is None:
                continue
            if runtime.is_morph_basis:
                offset_low, offset_high = hip_offset_range
            elif runtime.is_thrust_center:
                offset_low, offset_high = leg_offset_range
            else:
                continue
            if offset_high < offset_low:
                raise ValueError(
                    f"{runtime.spec.name} zero offset range must satisfy min <= max, "
                    f"got {(offset_low, offset_high)}"
                )
            runtime.zero_offset[env_ids] = self._sample_shared_plus_spread(
                env_ids, runtime.zero_offset, offset_low, offset_high, authority
            )

    def _randomize_observation_noise(self, env_ids: torch.Tensor, randomized: bool) -> None:
        """Draw this episode's estimator quality.

        Stored as a MULTIPLIER on each term's nominal scale rather than as an
        absolute value, so the per-term scales in M4EnvCfg stay the single place
        the nominal noise is described and the two cannot drift apart.
        """
        env = self.env
        multipliers = getattr(env, "_observation_noise_multiplier", None)
        if not multipliers:
            return
        ranges = {
            "pos_noise_scale": getattr(env.cfg, "pos_noise_scale_range", None),
            "lin_vel_noise_scale": getattr(env.cfg, "lin_vel_noise_scale_range", None),
        }
        enabled = randomized and bool(getattr(env.cfg, "randomize_observation_noise", False))
        for name, buffer in multipliers.items():
            scale_range = ranges.get(name)
            nominal = float(getattr(env.cfg, name, 0.0))
            if not enabled or scale_range is None or nominal <= 0.0:
                buffer[env_ids] = 1.0
                continue
            low, high = float(scale_range[0]), float(scale_range[1])
            if low < 0.0 or high < low:
                raise ValueError(f"{name}_range must satisfy 0 <= min <= max, got {(low, high)}")
            buffer[env_ids] = torch.empty_like(buffer[env_ids]).uniform_(low / nominal, high / nominal)

    def _reset_resilience_state(self, env_ids: torch.Tensor, randomized: bool) -> None:
        """Per-episode command dropout (Phase 5).

        Models the deployed stack's command stream stuttering, which the
        simulator otherwise hides.
        """
        env = self.env
        if not hasattr(env, "_command_dropout_remaining_steps"):
            return
        env._command_dropout_remaining_steps[env_ids] = 0
        if not randomized:
            env._command_dropout_probability[env_ids] = 0.0
            return
        env._command_dropout_probability[env_ids] = float(env.cfg.command_dropout_probability)


    def _com_setup(self):
        """Cache the articulation's nominal COMs and the base body row."""
        env = self.env
        if self._nominal_coms is not None:
            return True
        view = getattr(env._robot, "root_physx_view", None)
        if view is None or not hasattr(view, "set_coms"):
            return False
        self._nominal_coms = view.get_coms().clone()
        names = list(env._robot.body_names)
        base = env.vehicle.spec.base_body_name
        self._base_body_index = names.index(base) if base in names else 0

        # Apply the measured nominal offset to base_link's COM before any
        # per-episode band. Explicit body-frame vector: the previous
        # "shift toward the thrust center" also moved the CG vertically
        # (that direction was ~0.65 x, 0.76 z), which is not what the
        # measurement calls for.
        off = getattr(env.cfg, "com_nominal_offset_b", None)
        if off is not None and any(float(v) != 0.0 for v in off):
            b = self._base_body_index
            vec = torch.tensor([float(v) for v in off], dtype=self._nominal_coms.dtype)
            self._nominal_coms[:, b, :3] = self._nominal_coms[:, b, :3] + vec
            print(f"[com] nominal base_link COM offset applied: "
                  f"{[round(1000*float(v), 2) for v in off]} mm (body frame)")
        return True

    def _randomize_com(self, env_ids: torch.Tensor, randomized: bool) -> None:
        """Per-episode center-of-mass offset on the base body.

        set_coms takes the FULL (num_envs, num_bodies, 7) buffer (pos + quat), so
        the nominal must be written back for every other body and env -- a
        partial write silently zeroes the rows left out.
        """
        env = self.env
        if not bool(getattr(env.cfg, "randomize_com", False)):
            return
        if not self._com_setup():
            return
        coms = self._nominal_coms.clone()
        if randomized:
            ids = env_ids.detach().cpu()
            b = self._base_body_index
            lo_xy, hi_xy = env.cfg.com_offset_range_xy
            lo_z, hi_z = env.cfg.com_offset_range_z
            off = torch.empty(len(ids), 3, dtype=coms.dtype)
            off[:, 0].uniform_(lo_xy, hi_xy)
            off[:, 1].uniform_(lo_xy, hi_xy)
            off[:, 2].uniform_(lo_z, hi_z)
            coms[ids, b, :3] = self._nominal_coms[ids, b, :3] + off
        env._robot.root_physx_view.set_coms(
            coms, torch.arange(coms.shape[0], dtype=torch.int32)
        )

    def _reset_randomized(self, env_ids: torch.Tensor):
        env = self.env
        num_resets = len(env_ids)
        env._time_elapsed[env_ids] = 0.0
        disturbance_batch_scale, thrust_loss_batch_scale = self._sample_disturbance_batch_scale(num_resets)
        env._disturbance_batch_scale[env_ids] = disturbance_batch_scale
        env._thrust_loss_batch_scale[env_ids] = thrust_loss_batch_scale
        self.reset_task(env_ids, randomized=True)
        self._reset_drive_friction(env_ids, randomized=True)
        self._reset_actuators(env_ids, randomized=True)
        self._randomize_com(env_ids, randomized=True)
        self._randomize_observation_noise(env_ids, randomized=True)
        self._reset_resilience_state(env_ids, randomized=True)
        env.vehicle.reset_action_buffers(env_ids)
        disturbance_force_direction = self._sample_unit_vectors((num_resets, 1, 3))
        disturbance_moment_direction = self._sample_unit_vectors((num_resets, 1, 3))

        env._push_time[env_ids] = env.cfg.episode_length_s * torch.zeros_like(env._push_time[env_ids]).uniform_(
            0.0,
            0.5,
        )
        env._push_duration[env_ids] = torch.zeros_like(env._push_duration[env_ids]).uniform_(0.0, 0.2)
        env._push_end_time[env_ids] = env._push_time[env_ids] + env._push_duration[env_ids]

        disturbance_force_scale, disturbance_moment_scale = self._disturbance_scales()
        force_intensity = torch.zeros(num_resets, 1, 1, device=env.device).uniform_(
            -disturbance_force_scale,
            disturbance_force_scale,
        )
        moment_intensity = torch.zeros(num_resets, 1, 1, device=env.device).uniform_(
            -disturbance_moment_scale,
            disturbance_moment_scale,
        )
        env._disturbance_force[env_ids] = disturbance_batch_scale * force_intensity * disturbance_force_direction
        env._disturbance_moment[env_ids] = disturbance_batch_scale * moment_intensity * disturbance_moment_direction

        cts_force_scale, cts_moment_scale = self._continuous_disturbance_scales()
        cts_force_direction = self._sample_unit_vectors((num_resets, 1, 3))
        cts_moment_direction = self._sample_unit_vectors((num_resets, 1, 3))
        cts_force_intensity = torch.zeros(num_resets, 1, 1, device=env.device).uniform_(
            -cts_force_scale,
            cts_force_scale,
        )
        cts_moment_intensity = torch.zeros(num_resets, 1, 1, device=env.device).uniform_(
            -cts_moment_scale,
            cts_moment_scale,
        )
        env._disturbance_force_cts[env_ids] = disturbance_batch_scale * cts_force_intensity * cts_force_direction
        env._disturbance_moment_cts[env_ids] = disturbance_batch_scale * cts_moment_intensity * cts_moment_direction
        if env.cfg.randomize_motor_dynamics:
            tau_min, tau_max = env.actuator_tau_range()
            if tau_min <= 0.0 or tau_max < tau_min:
                raise ValueError(f"actuator tau range must satisfy 0 < min <= max, got {(tau_min, tau_max)}")
            tau_rise = torch.zeros_like(env._alpha_rise[env_ids]).uniform_(tau_min, tau_max)
            tau_fall = torch.zeros_like(env._alpha_fall[env_ids]).uniform_(tau_min, tau_max)
            env._actuator_tau_rise[env_ids] = tau_rise
            env._actuator_tau_fall[env_ids] = tau_fall
            env._alpha_rise[env_ids] = 1.0 - torch.exp(-float(env.step_dt) / tau_rise.clamp_min(1e-6))
            env._alpha_fall[env_ids] = 1.0 - torch.exp(-float(env.step_dt) / tau_fall.clamp_min(1e-6))
        else:
            env._actuator_tau_rise[env_ids] = env.cfg.T_m_0
            env._actuator_tau_fall[env_ids] = env.cfg.T_m_0
            env._alpha_rise[env_ids] = env.cfg.alpha_0
            env._alpha_fall[env_ids] = env.cfg.alpha_0


    def _reset_deterministic(self, env_ids: torch.Tensor):
        env = self.env
        num_resets = len(env_ids)
        env._time_elapsed[env_ids] = 0.0
        self.reset_task(env_ids, randomized=False)
        self._reset_drive_friction(env_ids, randomized=False)
        self._reset_actuators(env_ids, randomized=False)
        self._randomize_com(env_ids, randomized=False)
        self._randomize_observation_noise(env_ids, randomized=False)
        self._reset_resilience_state(env_ids, randomized=False)
        env.vehicle.reset_action_buffers(env_ids)
        disturbance_force_direction = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(num_resets, 1, 1)
        disturbance_force_direction = disturbance_force_direction / (
            torch.linalg.norm(disturbance_force_direction, dim=-1, keepdim=True) + 1e-6
        )
        disturbance_moment_direction = torch.normal(0.0, 1.0, size=(num_resets, 1, 3), device=env.device)
        disturbance_moment_direction = disturbance_moment_direction / (
            torch.linalg.norm(disturbance_moment_direction, dim=-1, keepdim=True) + 1e-6
        )

        env._push_time[env_ids] = 0.5 * torch.ones_like(env._push_time[env_ids])
        env._push_duration[env_ids] = 0.5 * torch.ones_like(env._push_duration[env_ids])
        env._push_end_time[env_ids] = env._push_time[env_ids] + env._push_duration[env_ids]
        env._disturbance_batch_scale[env_ids] = 1.0
        env._thrust_loss_batch_scale[env_ids] = 1.0

        disturbance_force_scale, disturbance_moment_scale = self._disturbance_scales()
        force_intensity = disturbance_force_scale
        moment_intensity = torch.normal(
            torch.zeros(num_resets, 1, 1, device=env.device),
            disturbance_moment_scale,
        )
        env._disturbance_force[env_ids] = force_intensity * disturbance_force_direction
        env._disturbance_moment[env_ids] = moment_intensity * disturbance_moment_direction

        cts_force_scale, cts_moment_scale = self._continuous_disturbance_scales()
        env._disturbance_force_cts[env_ids] = (
            torch.tensor([cts_force_scale, 0.0, 0.0], device=env.device)
            .repeat(
                num_resets,
                1,
            )
            .unsqueeze(1)
        )
        env._disturbance_moment_cts[env_ids] = (
            torch.tensor([0.0, 0.0, cts_moment_scale], device=env.device)
            .repeat(
                num_resets,
                1,
            )
            .unsqueeze(1)
        )

        env._actuator_tau_rise[env_ids] = env.cfg.T_m_0
        env._actuator_tau_fall[env_ids] = env.cfg.T_m_0
        env._alpha_rise[env_ids] = env.cfg.alpha_0
        env._alpha_fall[env_ids] = env.cfg.alpha_0
