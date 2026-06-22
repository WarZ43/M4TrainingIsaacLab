from __future__ import annotations

from numpy import pi
import torch

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_rotate


@configclass
class LandingTaskCfg:
    target_pos = [0.0, 0.0, 0.0]
    virtual_xy_offset_range = [15.0, 15.0]
    virtual_z_offset_range = [(0.8, 1.2), (0.8, 1.2)]
    initial_xy_range = [0.0, 0.5]
    initial_z_range = [(1.0, 2.0), (1.0, 2.0)]
    initial_roll_pitch_range = [pi / 6, pi / 6]
    initial_yaw_range = [pi, pi]

    contact_reward_requires_xy_by_stage = [False, False]
    contact_reward_xy_multiplier_by_stage = [False, True]
    contact_reward_accepts_invalid_by_stage = [True, False]
    invalid_contact_rew_factor = [0.25, 0.25]

    too_fast_vel = [2.0, 2.0]
    termination_dxy = [1.50, 3.50]
    termination_height = [3.0, 3.0]
    delta_d = [0.40, 0.60]

    vx_des, vy_des, vz_des = 0.0, 0.0, -0.50

    lin_vel_pen_scale = [-0.10, -0.10]
    ang_vel_pen_scale = [-0.30, -0.30]
    spin_pen_scale = [-0.30, -0.30]
    action_rate_pen_scale = [-0.80, -0.30]
    ground_thrust_pen_scale = [-0.13, -0.13]
    orientation_pen_scale = [-0.10, 0.00]
    yaw_angle_pen_scale = [-0.20, -0.20]

    impulse_pen = [-1.0, -1.00]
    died_pen = [-10.0, -10.0]
    timeout_pen = [-2.0, -32.0]

    distance_to_goal_xy_rew_scale = [2.0, 2.00]
    distance_to_goal_xy_rew_length = [0.25, 1.00]
    distance_to_goal_xy_progress_rew_scale = [0.0, 8.00]
    touchdown_distance_pen_scale = [0.0, -2.00]
    descending_rew_scale = [1.00, 0.60]
    tilt_rew_scale = [2.00, 0.80]
    contact_in_acceptance_rew_scale = [0.80, 4.00]
    touchdown_rew_scale = [0.00, 3.00]
    disturbance_force_scale = [2.0, 2.0]
    disturbance_moment_scale = [2.0, 2.0]
    disturbance_cts_force_scale = [1.0, 1.0]
    invalid_contact_termination_by_stage = [False, True]


class RewardMixer:
    def __init__(self, reward_keys: tuple[str, ...]):
        self.reward_keys = reward_keys

    def sum(self, rewards: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sum(torch.stack([rewards[key] for key in self.reward_keys]), dim=0)


class LandingTask:
    """Landing-specific curriculum, rewards, termination, and metrics."""

    reward_keys = (
        "lin_vel_pen",
        "ang_vel_pen",
        "action_rate_pen",
        "ground_thrust_penalty",
        "spin_penalty",
        "orientation_pen",
        "yaw_angle_pen",
        "died_penalty",
        "impulse_penalty",
        "distance_to_goal_xy_rew",
        "distance_to_goal_xy_progress_rew",
        "descending_rew",
        "tilt_rew",
        "touchdown_rew",
        "touchdown_distance_penalty",
        "contact_in_acceptance_rew",
        "timeout_high_penalty",
    )

    def __init__(self, env, cfg: LandingTaskCfg):
        self.env = env
        self.cfg = cfg
        self.reward_mixer = RewardMixer(self.reward_keys)

        env._desired_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._target_pos_obs_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._virtual_xy_offset_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._current_contacts = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._in_acceptance_ball = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._in_acceptance_xy = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._first_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ep_contact_in_acceptance = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._initial_yaw = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        env._previous_distance_to_goal_xy = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

        env._lin_vel_des = torch.zeros(3, device=env.device)
        env._lin_vel_des[0] = cfg.vx_des
        env._lin_vel_des[1] = cfg.vy_des
        env._lin_vel_des[2] = cfg.vz_des

        env._episode_sums = {
            key: torch.zeros(env.num_envs, dtype=torch.float, device=env.device) for key in self.reward_keys
        }

    def stage_value(self, values, name: str):
        try:
            stage = int(self.env.cfg.curriculum_stage)
            if stage < 1:
                raise ValueError(f"curriculum_stage must be >= 1, got {self.env.cfg.curriculum_stage}")
            stage_idx = stage - 1
            return values[stage_idx]
        except TypeError:
            return values
        except IndexError as exc:
            raise ValueError(f"curriculum_stage={self.env.cfg.curriculum_stage} has no entry for {name}") from exc

    def disturbance_force_scale(self) -> float:
        return self.env.cfg.disturbance_force_scale * self.stage_value(
            self.cfg.disturbance_force_scale,
            "disturbance_force_scale",
        )

    def disturbance_moment_scale(self) -> float:
        return self.env.cfg.disturbance_moment_scale * self.stage_value(
            self.cfg.disturbance_moment_scale,
            "disturbance_moment_scale",
        )

    def disturbance_cts_force_scale(self) -> float:
        return self.env.cfg.disturbance_force_scale * self.stage_value(
            self.cfg.disturbance_cts_force_scale,
            "disturbance_cts_force_scale",
        )

    def reset_initial_state(self, env_ids: torch.Tensor, randomized: bool):
        env = self.env
        num_resets = len(env_ids)
        target = torch.tensor(self.cfg.target_pos, device=env.device, dtype=torch.float).repeat(num_resets, 1)
        virtual_offset = torch.zeros(num_resets, 3, device=env.device)

        if randomized:
            virtual_offset[:, :2] = torch.zeros(num_resets, 2, device=env.device).uniform_(
                -self.stage_value(self.cfg.virtual_xy_offset_range, "virtual_xy_offset_range"),
                self.stage_value(self.cfg.virtual_xy_offset_range, "virtual_xy_offset_range"),
            )
            virtual_z_offset_min, virtual_z_offset_max = self.stage_value(
                self.cfg.virtual_z_offset_range,
                "virtual_z_offset_range",
            )
            virtual_offset[:, 2] = torch.zeros(num_resets, device=env.device).uniform_(
                virtual_z_offset_min,
                virtual_z_offset_max,
            )

            root_state = env._robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += env._terrain.env_origins[env_ids]
            initial_xy_offset = torch.zeros(num_resets, 2, device=env.device).uniform_(
                -self.stage_value(self.cfg.initial_xy_range, "initial_xy_range"),
                self.stage_value(self.cfg.initial_xy_range, "initial_xy_range"),
            )
            initial_z_min, initial_z_max = self.stage_value(
                self.cfg.initial_z_range,
                "initial_z_range",
            )
            root_state[:, :2] += initial_xy_offset
            root_state[:, 2] = torch.zeros_like(root_state[:, 2]).uniform_(
                initial_z_min,
                initial_z_max,
            )
            root_state[:, 2] += env._terrain.env_origins[env_ids, 2]
            roll_pitch_range = self.stage_value(self.cfg.initial_roll_pitch_range, "initial_roll_pitch_range")
            yaw_range = self.stage_value(self.cfg.initial_yaw_range, "initial_yaw_range")
            roll = torch.zeros(num_resets, device=env.device).uniform_(-roll_pitch_range, roll_pitch_range)
            pitch = torch.zeros(num_resets, device=env.device).uniform_(-roll_pitch_range, roll_pitch_range)
            yaw = torch.zeros(num_resets, device=env.device).uniform_(-yaw_range, yaw_range)
            root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
            root_state[:, 7:10] = torch.zeros_like(root_state[:, 7:10]).uniform_(
                env.cfg.initial_lin_vel_range[0],
                env.cfg.initial_lin_vel_range[1],
            )
            root_state[:, 10:13] = torch.zeros_like(root_state[:, 10:13]).uniform_(
                env.cfg.initial_ang_vel_range[0],
                env.cfg.initial_ang_vel_range[1],
            )
            joint_pos, joint_vel = env.vehicle.randomized_joint_state(env_ids)
        else:
            root_state = env._robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += env._terrain.env_origins[env_ids]
            joint_pos, joint_vel = env.vehicle.deterministic_joint_state(env_ids)

        env._desired_pos_w[env_ids] = env._terrain.env_origins[env_ids] + target
        env._virtual_xy_offset_w[env_ids] = virtual_offset
        env._target_pos_obs_w[env_ids] = env._desired_pos_w[env_ids] + virtual_offset
        env._initial_yaw[env_ids] = self._yaw_from_quat(root_state[:, 3:7])
        env._previous_distance_to_goal_xy[env_ids] = torch.linalg.norm(
            env._desired_pos_w[env_ids, :2] - root_state[:, :2],
            dim=1,
        )

        env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        env._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        env._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)

    def get_rewards(self) -> torch.Tensor:
        env = self.env
        died, time_out = self.get_dones()
        too_fast_vel = self.stage_value(self.cfg.too_fast_vel, "too_fast_vel")

        valid_contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._valid_contact_ids]
        valid_contacts = torch.any(valid_contact_time > 0.0, dim=1)
        invalid_contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._invalid_contact_ids]
        invalid_contacts = torch.any(invalid_contact_time > 0.0, dim=1)
        if self.stage_value(
            self.cfg.contact_reward_accepts_invalid_by_stage,
            "contact_reward_accepts_invalid_by_stage",
        ):
            env._current_contacts = valid_contacts | invalid_contacts
        else:
            env._current_contacts = valid_contacts
        new_contacts = torch.logical_and(
            torch.logical_xor(env._current_contacts, env._first_contact),
            env._current_contacts,
        )
        new_contact_idx = torch.nonzero(new_contacts)
        env._first_contact[new_contact_idx] = new_contacts[new_contact_idx]

        distance_to_goal_xy = torch.linalg.norm(
            env._desired_pos_w[:, :2] - env._robot.data.root_link_pos_w[:, :2],
            dim=1,
        )
        distance_to_goal_xy_rew_length = max(
            float(self.stage_value(self.cfg.distance_to_goal_xy_rew_length, "distance_to_goal_xy_rew_length")),
            1e-6,
        )
        distance_to_goal_xy_mapped = torch.exp(-distance_to_goal_xy / distance_to_goal_xy_rew_length)
        distance_to_goal_xy_progress = env._previous_distance_to_goal_xy - distance_to_goal_xy
        distance_to_goal_xy_progress = distance_to_goal_xy_progress * (~env._first_contact).float()
        env._previous_distance_to_goal_xy[:] = distance_to_goal_xy.detach()

        lin_vel = torch.sum(torch.square(env._robot.data.root_com_lin_vel_w), dim=1)
        ang_vel = torch.sum(torch.square(env._robot.data.root_com_ang_vel_b[:, :2]), dim=1)
        spin_vel = torch.square(env._robot.data.root_com_ang_vel_b[:, 2])

        descending_error = torch.square(env._robot.data.root_com_lin_vel_w[:, 2] - env._lin_vel_des[2]) * (
            ~env._first_contact
        )
        descending_error_mapped = torch.exp(-descending_error / 0.25)

        tilt = env.vehicle.landing_tuck_position()
        tuck_target = env.vehicle.landing_tuck_target()
        tilt_error = torch.square(tilt - tuck_target)
        tilt_error_mapped = torch.exp(-tilt_error / 0.25)

        env._in_acceptance_xy = (distance_to_goal_xy - self.stage_value(self.cfg.delta_d, "delta_d")) < 0.0

        action_rate_flight = env.vehicle.rotor_action_rate()
        ground_thrust = env.vehicle.rotor_action_magnitude() * env._first_contact

        flat_orientation = torch.abs(1 - self._quat_axis(env._robot.data.root_link_quat_w, 2)[..., 2])
        yaw_error = self._wrap_to_pi(self._yaw_from_quat(env._robot.data.root_link_quat_w) - env._initial_yaw)

        accepted_contact = env._current_contacts & ~died
        if self.stage_value(self.cfg.contact_reward_requires_xy_by_stage, "contact_reward_requires_xy_by_stage"):
            accepted_contact &= env._in_acceptance_xy
        new_valid_contact = new_contacts & accepted_contact
        has_landed = env._ep_contact_in_acceptance | accepted_contact
        env._ep_contact_in_acceptance = has_landed

        timeout_no_contact_penalty = (
            time_out.float() * (~has_landed).float() * self.stage_value(self.cfg.timeout_pen, "timeout_pen")
        )

        excess_vel = torch.clamp(torch.linalg.norm(env._robot.data.root_com_lin_vel_w, dim=1) - too_fast_vel, min=0.0)

        tuck_error = torch.square(tilt - tuck_target)
        tuck_multiplier = torch.exp(-tuck_error / 0.5)
        invalid_contact_factor = self.stage_value(
            self.cfg.invalid_contact_rew_factor,
            "invalid_contact_rew_factor",
        )
        contact_rew_factor = torch.where(
            valid_contacts,
            tuck_multiplier,
            invalid_contact_factor * torch.ones_like(tuck_multiplier),
        )
        contact_xy_multiplier = torch.ones_like(distance_to_goal_xy)
        if self.stage_value(
            self.cfg.contact_reward_xy_multiplier_by_stage,
            "contact_reward_xy_multiplier_by_stage",
        ):
            contact_xy_scale = max(float(self.stage_value(self.cfg.delta_d, "delta_d")), 1e-6)
            contact_xy_multiplier = torch.exp(-distance_to_goal_xy / contact_xy_scale)

        gated_contact_rew = (
            accepted_contact.float()
            * contact_rew_factor
            * contact_xy_multiplier
            * self.stage_value(self.cfg.contact_in_acceptance_rew_scale, "contact_in_acceptance_rew_scale")
            * env.step_dt
        )
        touchdown_rew = (
            new_valid_contact.float()
            * self.stage_value(self.cfg.touchdown_rew_scale, "touchdown_rew_scale")
            * (0.25 + 0.75 * tuck_multiplier)
            * contact_xy_multiplier
        )
        touchdown_distance_penalty = (
            new_valid_contact.float()
            * distance_to_goal_xy
            * self.stage_value(self.cfg.touchdown_distance_pen_scale, "touchdown_distance_pen_scale")
        )

        rewards = {
            "lin_vel_pen": lin_vel * self.stage_value(self.cfg.lin_vel_pen_scale, "lin_vel_pen_scale") * env.step_dt,
            "ang_vel_pen": ang_vel * self.stage_value(self.cfg.ang_vel_pen_scale, "ang_vel_pen_scale") * env.step_dt,
            "action_rate_pen": action_rate_flight
            * self.stage_value(self.cfg.action_rate_pen_scale, "action_rate_pen_scale")
            * env.step_dt,
            "ground_thrust_penalty": ground_thrust
            * self.stage_value(self.cfg.ground_thrust_pen_scale, "ground_thrust_pen_scale")
            * env.step_dt,
            "spin_penalty": spin_vel * self.stage_value(self.cfg.spin_pen_scale, "spin_pen_scale") * env.step_dt,
            "orientation_pen": flat_orientation
            * self.stage_value(self.cfg.orientation_pen_scale, "orientation_pen_scale")
            * env.step_dt,
            "yaw_angle_pen": torch.square(yaw_error)
            * self.stage_value(self.cfg.yaw_angle_pen_scale, "yaw_angle_pen_scale")
            * env.step_dt,
            "died_penalty": died * (self.stage_value(self.cfg.died_pen, "died_pen") - (excess_vel * 10.0)),
            "impulse_penalty": env._current_impulse.squeeze(dim=1)
            * self.stage_value(self.cfg.impulse_pen, "impulse_pen"),
            "distance_to_goal_xy_rew": distance_to_goal_xy_mapped
            * self.stage_value(self.cfg.distance_to_goal_xy_rew_scale, "distance_to_goal_xy_rew_scale")
            * env.step_dt,
            "distance_to_goal_xy_progress_rew": distance_to_goal_xy_progress
            * self.stage_value(
                self.cfg.distance_to_goal_xy_progress_rew_scale,
                "distance_to_goal_xy_progress_rew_scale",
            ),
            "descending_rew": descending_error_mapped
            * self.stage_value(self.cfg.descending_rew_scale, "descending_rew_scale")
            * env.step_dt,
            "tilt_rew": tilt_error_mapped * self.stage_value(self.cfg.tilt_rew_scale, "tilt_rew_scale") * env.step_dt,
            "touchdown_rew": touchdown_rew,
            "touchdown_distance_penalty": touchdown_distance_penalty,
            "contact_in_acceptance_rew": gated_contact_rew,
            "timeout_high_penalty": timeout_no_contact_penalty,
        }
        reward = self.reward_mixer.sum(rewards)

        for key, value in rewards.items():
            env._episode_sums[key] += value
        return reward

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        time_out = env.episode_length_buf >= env.max_episode_length - 1
        died1 = torch.linalg.norm(env._robot.data.root_com_lin_vel_w, dim=1) > self.stage_value(
            self.cfg.too_fast_vel,
            "too_fast_vel",
        )
        died2 = torch.linalg.norm(
            env._desired_pos_w[:, :2] - env._robot.data.root_link_pos_w[:, :2],
            dim=1,
        ) > self.stage_value(self.cfg.termination_dxy, "termination_dxy")
        died3 = env._robot.data.root_link_pos_w[:, 2] > self.stage_value(
            self.cfg.termination_height,
            "termination_height",
        )

        invalid_contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._invalid_contact_ids]
        invalid_contact_termination = self.stage_value(
            self.cfg.invalid_contact_termination_by_stage,
            "invalid_contact_termination_by_stage",
        )
        if invalid_contact_termination:
            died4 = torch.any(invalid_contact_time > 0.0, dim=1)
        else:
            died4 = torch.zeros_like(died1)

        if env.cfg.terminate:
            died = died1 | died2 | died3 | died4
        else:
            died = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        return died, time_out

    def log_episode(self, env_ids: torch.Tensor):
        env = self.env
        final_distance_to_goal = torch.linalg.norm(
            env._desired_pos_w[env_ids] - env._robot.data.root_link_pos_w[env_ids],
            dim=1,
        ).mean()

        extras = {}
        for key in env._episode_sums.keys():
            episodic_sum_avg = self._finite_mean(env._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = torch.nan_to_num(
                episodic_sum_avg / env.max_episode_length_s,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            env._episode_sums[key][env_ids] = 0.0

        env.extras["log"] = {}
        env.extras["log"].update(extras)

        extras = {}
        extras["Metrics/final_distance_to_goal"] = torch.nan_to_num(
            final_distance_to_goal,
            nan=0.0,
            posinf=1e6,
            neginf=1e6,
        ).item()
        extras["Metrics/distance_to_goal_epoch_av"] = float(
            torch.nan_to_num(
                torch.tensor(env.distance_to_goal_epoch_av, device=env.device),
                nan=0.0,
                posinf=1e6,
                neginf=1e6,
            ).item()
        )

        landed = env._ep_contact_in_acceptance[env_ids]
        timeout = env.reset_time_outs[env_ids]
        num_resets = max(len(env_ids), 1)

        extras["Landing/contact_acceptance_frequency"] = torch.count_nonzero(landed & timeout).item() / num_resets
        extras["Landing/timeout_with_contact_frequency"] = torch.count_nonzero(timeout & landed).item() / num_resets
        extras["Landing/timeout_no_contact_frequency"] = torch.count_nonzero(timeout & ~landed).item() / num_resets
        extras["Landing/terminated_no_contact_frequency"] = (
            torch.count_nonzero(env.reset_terminated[env_ids] & ~landed).item() / num_resets
        )
        env.extras["log"].update(extras)

    def reset_episode_state(self, env_ids: torch.Tensor):
        env = self.env
        env._ep_contact_in_acceptance[env_ids] = False
        env._first_contact[env_ids] = False
        env._current_contacts[env_ids] = False
        env._in_acceptance_ball[env_ids] = False
        env._in_acceptance_xy[env_ids] = False

    @staticmethod
    def _finite_mean(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros((), device=values.device)
        return torch.nan_to_num(torch.mean(values), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _quat_axis(q: torch.Tensor, axis: int = 0) -> torch.Tensor:
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)

    @staticmethod
    def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q.unbind(dim=1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))
