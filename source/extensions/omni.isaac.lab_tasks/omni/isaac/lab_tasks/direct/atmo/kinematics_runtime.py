from __future__ import annotations

import torch


class KinematicsRuntime:
    """Kinematics and geometry shared by the concrete drone layouts."""
    def _body_com_pos_w(self) -> torch.Tensor:
        return self.env._robot.data.body_com_pos_w

    def _body_quat_w(self, body_ids: list[int]) -> torch.Tensor:
        return self.env._robot.data.body_link_quat_w[:, body_ids, :]
    def _current_rotor_thrust_axis_w(self, joint_positions: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        env = self.env
        if joint_positions is not None and self.spec.morph_joint_group in self.joint_groups:
            return self._rotate_link_vectors_to_world(
                self._base_quat_w().expand(-1, self.rotor_count, -1),
                self._morph_tilt_thrust_axis_body(joint_positions),
            )
        thrust_axis_link = self.thrust_axis.unsqueeze(0).expand(env.num_envs, -1, -1)
        return self._rotate_link_vectors_to_world(self._body_quat_w(self.rotor_axis_ids), thrust_axis_link)
    def _morph_tilt_thrust_axis_body(self, joint_positions: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        env = self.env
        rotor_tilt = self._morph_tilt_per_rotor(joint_positions).clamp(0.0, torch.pi / 2).unsqueeze(-1)
        rotor_roll = self.rotor_roll_sign.reshape(1, self.rotor_count, 1) * rotor_tilt
        axis_x = torch.zeros_like(rotor_roll)
        # Start at +Z. Right rotors use clockwise +roll; left rotors use counterclockwise -roll.
        axis_y = -torch.sin(rotor_roll)
        axis_z = torch.cos(rotor_roll)
        return torch.cat((axis_x, axis_y, axis_z), dim=-1)
    def _morph_tilt_per_rotor(self, joint_positions: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        env = self.env
        group_name = self.spec.morph_joint_group
        if group_name is None or group_name not in self.joint_groups:
            return torch.zeros(env.num_envs, self.rotor_count, device=env.device)

        tilt = (
            joint_positions[group_name]
            if joint_positions is not None and group_name in joint_positions
            else self.joint_group_positions(group_name)
        )
        if tilt.shape[1] == 1:
            return tilt.repeat(1, self.rotor_count)
        if tilt.shape[1] == self.rotor_count:
            rotor_indices = torch.as_tensor(
                [rotor.action_index for rotor in self.spec.rotors],
                device=env.device,
                dtype=torch.long,
            )
            return tilt[:, rotor_indices]
        raise ValueError(
            f"{self.spec.name} morph-tilt group '{group_name}' has width {tilt.shape[1]}, "
            f"but expected 1 or rotor_count={self.rotor_count}"
        )
    @staticmethod
    def _rotate_link_vectors_to_world(quats_w: torch.Tensor, vectors_b: torch.Tensor) -> torch.Tensor:
        q_vec = quats_w[..., 1:4]
        q_w = quats_w[..., 0:1]
        uv = torch.cross(q_vec, vectors_b, dim=-1)
        uuv = torch.cross(q_vec, uv, dim=-1)
        return vectors_b + 2.0 * (q_w * uv + uuv)
    @staticmethod
    def _rotate_world_vectors_to_body(quats_w: torch.Tensor, vectors_w: torch.Tensor) -> torch.Tensor:
        q_vec = -quats_w[..., 1:4]
        q_w = quats_w[..., 0:1]
        uv = torch.cross(q_vec, vectors_w, dim=-1)
        uuv = torch.cross(q_vec, uv, dim=-1)
        return vectors_w + 2.0 * (q_w * uv + uuv)
    def _thrust_center_xy_body(self) -> torch.Tensor:
        env = self.env
        rotor_r_b = self._rotor_relative_pos_body()
        return self._weighted_rotor_center_xy(rotor_r_b, env.kT)
    @staticmethod
    def _weighted_rotor_center_xy(rotor_r_b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        safe_weights = weights.clamp_min(0.0)
        center_b = torch.sum(rotor_r_b * safe_weights.unsqueeze(-1), dim=1) / torch.sum(
            safe_weights,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        return center_b[:, :2]
    def _rotor_relative_pos_body(self, joint_positions: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if joint_positions is None:
            cached = self._observation_pass_cache.get("rotor_r_b")
            if cached is not None:
                return cached
        if self._use_leg_geometry():
            rotor_r_b = self._leg_geometry_rotor_relative_pos_body(joint_positions)
            if joint_positions is None:
                self._observation_pass_cache["rotor_r_b"] = rotor_r_b
            return rotor_r_b
        rotor_r_w = self._rotor_relative_pos_world()
        base_quat_w = self._base_quat_w().expand(-1, self.rotor_count, -1)
        rotor_r_b = self._rotate_world_vectors_to_body(base_quat_w, rotor_r_w)
        if joint_positions is None:
            self._observation_pass_cache["rotor_r_b"] = rotor_r_b
        return rotor_r_b
    def _use_leg_geometry(self) -> bool:
        return (
            self.spec.leg_joint_group is not None
            and self.spec.morph_joint_group is not None
            and self.spec.morph_joint_group in self.joint_groups
            and self.spec.leg_joint_group in self.joint_groups
        )
    def _leg_geometry_rotor_relative_pos_body(self, joint_positions: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        leg_runtime = self.joint_groups[self.spec.leg_joint_group]
        leg_positions = (
            joint_positions[self.spec.leg_joint_group]
            if joint_positions is not None and self.spec.leg_joint_group in joint_positions
            else self.joint_group_positions(leg_runtime.spec.name)
        )
        return self._leg_geometry_rotor_relative_pos_body_for_leg_positions(leg_positions, joint_positions)
    def _leg_geometry_rotor_relative_pos_body_for_leg_positions(
        self,
        leg_positions: torch.Tensor,
        joint_positions: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        env = self.env
        leg_runtime = self.joint_groups[self.spec.leg_joint_group]
        leg_positions = leg_positions.to(device=env.kT.device, dtype=env.kT.dtype)
        leg_q = self._joint_group_position_to_sim(leg_runtime, leg_positions).to(dtype=env.kT.dtype)
        return self._leg_geometry_rotor_relative_pos_body_from_leg_q(leg_q, joint_positions)
    def _leg_geometry_rotor_relative_pos_body_from_leg_q(
        self,
        leg_q: torch.Tensor,
        joint_positions: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        env = self.env
        geom = self._leg_geometry_constants(env.kT.device, env.kT.dtype)
        morph_runtime = self.joint_groups[self.spec.morph_joint_group]
        if joint_positions is not None and self.spec.morph_joint_group in joint_positions:
            hip_q = self._joint_group_position_to_sim(
                morph_runtime, joint_positions[self.spec.morph_joint_group]
            ).to(dtype=env.kT.dtype)
        else:
            hip_q = env._robot.data.joint_pos[:, morph_runtime.joint_ids].to(dtype=env.kT.dtype)
        if hip_q.shape[1] == 1:
            hip_q = hip_q.expand(-1, self.rotor_count)
        hip_rot = torch.matmul(
            geom["hip_rot"].unsqueeze(0),
            self._z_rotation_matrix(hip_q * geom["hip_axis_sign"]),
        )
        leg_joint_pos = geom["hip_xyz"].unsqueeze(0) + torch.matmul(
            hip_rot,
            geom["leg_xyz"].reshape(1, self.rotor_count, 3, 1),
        ).squeeze(-1)
        leg_parent_rot = torch.matmul(hip_rot, geom["leg_rot"].unsqueeze(0))
        leg_rot = torch.matmul(
            leg_parent_rot,
            self._z_rotation_matrix(leg_q * geom["leg_axis_sign"]),
        )
        rotor_pos_b = leg_joint_pos + torch.matmul(
            leg_rot,
            geom["blade_xyz"].reshape(1, self.rotor_count, 3, 1),
        ).squeeze(-1)
        return rotor_pos_b - geom["base_com"].reshape(1, 1, 3)
    def _leg_geometry_constants(self, device, dtype) -> dict[str, torch.Tensor]:
        if self._leg_geometry_cache is not None:
            return self._leg_geometry_cache
        geometry = self.spec.leg_geometry
        if geometry is None:
            raise ValueError(f"{self.spec.name} requires leg_geometry for its configured leg joint group")
        hip_xyz = torch.as_tensor(geometry.hip_xyz, device=device, dtype=dtype)
        leg_xyz = torch.as_tensor(geometry.leg_xyz, device=device, dtype=dtype)
        blade_xyz = torch.as_tensor(geometry.blade_xyz, device=device, dtype=dtype)
        constants = {
            "base_com": torch.as_tensor(geometry.base_com, device=device, dtype=dtype),
            "hip_xyz": hip_xyz,
            "hip_rot": self._rpy_matrix(torch.as_tensor(geometry.hip_rpy, device=device, dtype=dtype)),
            "hip_axis_sign": torch.as_tensor(geometry.hip_axis_sign, device=device, dtype=dtype).reshape(1, -1),
            "leg_xyz": leg_xyz,
            "leg_rot": self._rpy_matrix(torch.as_tensor(geometry.leg_rpy, device=device, dtype=dtype)),
            "leg_axis_sign": torch.as_tensor(geometry.leg_axis_sign, device=device, dtype=dtype).reshape(1, -1),
            "blade_xyz": blade_xyz,
        }
        if self.rotor_count != hip_xyz.shape[0]:
            raise ValueError(
                f"leg geometry has {hip_xyz.shape[0]} corners but the vehicle has {self.rotor_count} rotors"
            )
        self._leg_geometry_cache = constants
        return constants
    @staticmethod
    def _z_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        rot = torch.zeros(*angle.shape, 3, 3, device=angle.device, dtype=angle.dtype)
        rot[..., 0, 0] = cos
        rot[..., 0, 1] = -sin
        rot[..., 1, 0] = sin
        rot[..., 1, 1] = cos
        rot[..., 2, 2] = 1.0
        return rot
    @staticmethod
    def _rpy_matrix(rpy: torch.Tensor) -> torch.Tensor:
        roll = rpy[..., 0]
        pitch = rpy[..., 1]
        yaw = rpy[..., 2]
        sr = torch.sin(roll)
        cr = torch.cos(roll)
        sp = torch.sin(pitch)
        cp = torch.cos(pitch)
        sy = torch.sin(yaw)
        cy = torch.cos(yaw)
        rot = torch.empty(*rpy.shape[:-1], 3, 3, device=rpy.device, dtype=rpy.dtype)
        rot[..., 0, 0] = cy * cp
        rot[..., 0, 1] = cy * sp * sr - sy * cr
        rot[..., 0, 2] = cy * sp * cr + sy * sr
        rot[..., 1, 0] = sy * cp
        rot[..., 1, 1] = sy * sp * sr + cy * cr
        rot[..., 1, 2] = sy * sp * cr - cy * sr
        rot[..., 2, 0] = -sp
        rot[..., 2, 1] = cp * sr
        rot[..., 2, 2] = cp * cr
        return rot
    def _rotor_relative_pos_world(self) -> torch.Tensor:
        cached = self._observation_pass_cache.get("rotor_r_w")
        if cached is not None:
            return cached
        body_com_pos_w = self._body_com_pos_w()
        rotor_r_w = body_com_pos_w[:, self.rotor_ids, :] - body_com_pos_w[:, self.base_link, :].unsqueeze(1)
        self._observation_pass_cache["rotor_r_w"] = rotor_r_w
        return rotor_r_w
    def _base_quat_w(self) -> torch.Tensor:
        cached = self._observation_pass_cache.get("base_quat_w")
        if cached is not None:
            return cached
        base_quat_w = self._body_quat_w([self.base_link])
        self._observation_pass_cache["base_quat_w"] = base_quat_w
        return base_quat_w
    def _morph_trig_observation(self) -> torch.Tensor:
        values = []
        for group_name, runtime in self.joint_groups.items():
            if runtime.spec.action_mapping not in {"direct", "morph_tilt_basis", "thrust_center_shift"}:
                continue
            values.append(self._joint_group_position_observation(group_name))
        if not values:
            return torch.empty(self.env.num_envs, 0, device=self.env.device)
        angles = torch.cat(values, dim=1)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    def _joint_group_position_observation(self, group_name: str) -> torch.Tensor:
        cache_key = f"joint_pos:{group_name}"
        cached = self._observation_pass_cache.get(cache_key)
        if cached is not None:
            return cached
        runtime = self.joint_groups[group_name]
        joint_pos = self.env._robot.data.joint_pos[:, runtime.joint_ids]
        value = self._joint_group_position_from_sim(runtime, joint_pos)
        self._observation_pass_cache[cache_key] = value
        return value
