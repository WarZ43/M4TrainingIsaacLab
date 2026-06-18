from __future__ import annotations

import torch

from omni.isaac.lab.utils.math import quat_rotate


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
        "died_penalty",
        "impulse_penalty",
        "distance_to_goal_xy_rew",
        "descending_rew",
        "tilt_rew",
        "touchdown_rew",
        "contact_in_acceptance_rew",
        "timeout_high_penalty",
    )

    def __init__(self, env):
        self.env = env
        self.reward_mixer = RewardMixer(self.reward_keys)

        env._desired_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._current_contacts = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._in_acceptance_ball = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._in_acceptance_xy = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._first_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ep_contact_in_acceptance = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        env._lin_vel_des = torch.zeros(3, device=env.device)
        env._lin_vel_des[0] = env.cfg.vx_des
        env._lin_vel_des[1] = env.cfg.vy_des
        env._lin_vel_des[2] = env.cfg.vz_des

        env._episode_sums = {
            key: torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
            for key in self.reward_keys
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
            raise ValueError(
                f"curriculum_stage={self.env.cfg.curriculum_stage} has no entry for {name}"
            ) from exc

    def get_rewards(self) -> torch.Tensor:
        env = self.env
        died, time_out = self.get_dones()
        too_fast_vel = self.stage_value(env.cfg.too_fast_vel, "too_fast_vel")

        current_contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._valid_contact_ids]
        num_contact = torch.sum(current_contact_time > 0.0, dim=1)
        env._current_contacts = num_contact > 0
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
        distance_to_goal_xy_mapped = torch.exp(-distance_to_goal_xy / 0.25)

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

        env._in_acceptance_xy = (distance_to_goal_xy - self.stage_value(env.cfg.delta_d, "delta_d")) < 0.0

        action_rate_flight = env.vehicle.rotor_action_rate()
        ground_thrust = env.vehicle.rotor_action_magnitude() * env._first_contact

        flat_orientation = torch.abs(1 - self._quat_axis(env._robot.data.root_link_quat_w, 2)[..., 2])

        accepted_contact = env._current_contacts & ~died
        if self.stage_value(env.cfg.contact_reward_requires_xy_by_stage, "contact_reward_requires_xy_by_stage"):
            accepted_contact &= env._in_acceptance_xy
        new_valid_contact = new_contacts & accepted_contact
        has_landed = env._ep_contact_in_acceptance | accepted_contact
        env._ep_contact_in_acceptance = has_landed

        timeout_no_contact_penalty = (
            time_out.float()
            * (~has_landed).float()
            * self.stage_value(env.cfg.timeout_pen, "timeout_pen")
        )

        excess_vel = torch.clamp(torch.linalg.norm(env._robot.data.root_com_lin_vel_w, dim=1) - too_fast_vel, min=0.0)

        tuck_error = torch.square(tilt - tuck_target)
        tuck_multiplier = torch.exp(-tuck_error / 0.5)

        gated_contact_rew = (
            accepted_contact.float()
            * tuck_multiplier
            * self.stage_value(env.cfg.contact_in_acceptance_rew_scale, "contact_in_acceptance_rew_scale")
            * env.step_dt
        )
        touchdown_rew = (
            new_valid_contact.float()
            * self.stage_value(env.cfg.touchdown_rew_scale, "touchdown_rew_scale")
            * (0.25 + 0.75 * tuck_multiplier)
        )

        rewards = {
            "lin_vel_pen": lin_vel * self.stage_value(env.cfg.lin_vel_pen_scale, "lin_vel_pen_scale") * env.step_dt,
            "ang_vel_pen": ang_vel * self.stage_value(env.cfg.ang_vel_pen_scale, "ang_vel_pen_scale") * env.step_dt,
            "action_rate_pen": action_rate_flight
            * self.stage_value(env.cfg.action_rate_pen_scale, "action_rate_pen_scale")
            * env.step_dt,
            "ground_thrust_penalty": ground_thrust
            * self.stage_value(env.cfg.ground_thrust_pen_scale, "ground_thrust_pen_scale")
            * env.step_dt,
            "spin_penalty": spin_vel * self.stage_value(env.cfg.spin_pen_scale, "spin_pen_scale") * env.step_dt,
            "orientation_pen": flat_orientation
            * self.stage_value(env.cfg.orientation_pen_scale, "orientation_pen_scale")
            * env.step_dt,
            "died_penalty": died * (self.stage_value(env.cfg.died_pen, "died_pen") - (excess_vel * 10.0)),
            "impulse_penalty": env._current_impulse.squeeze(dim=1)
            * self.stage_value(env.cfg.impulse_pen, "impulse_pen"),
            "distance_to_goal_xy_rew": distance_to_goal_xy_mapped
            * self.stage_value(env.cfg.distance_to_goal_xy_rew_scale, "distance_to_goal_xy_rew_scale")
            * env.step_dt,
            "descending_rew": descending_error_mapped
            * self.stage_value(env.cfg.descending_rew_scale, "descending_rew_scale")
            * env.step_dt,
            "tilt_rew": tilt_error_mapped * self.stage_value(env.cfg.tilt_rew_scale, "tilt_rew_scale") * env.step_dt,
            "touchdown_rew": touchdown_rew,
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
            env.cfg.too_fast_vel,
            "too_fast_vel",
        )
        died2 = torch.linalg.norm(
            env._desired_pos_w[:, :2] - env._robot.data.root_link_pos_w[:, :2],
            dim=1,
        ) > self.stage_value(env.cfg.termination_dxy, "termination_dxy")
        died3 = env._robot.data.root_link_pos_w[:, 2] > self.stage_value(
            env.cfg.termination_height,
            "termination_height",
        )

        invalid_contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._invalid_contact_ids]
        died4 = torch.any(invalid_contact_time > 0.0, dim=1)

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
            episodic_sum_avg = torch.mean(env._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / env.max_episode_length_s
            env._episode_sums[key][env_ids] = 0.0

        env.extras["log"] = {}
        env.extras["log"].update(extras)

        extras = {}
        extras["Episode_Termination/died"] = torch.count_nonzero(env.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(env.reset_time_outs[env_ids]).item()
        extras["Curriculum/stage"] = int(env.cfg.curriculum_stage)
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/distance_to_goal_epoch_av"] = env.distance_to_goal_epoch_av
        extras["Metrics/final_height"] = env._robot.data.root_link_pos_w[env_ids, 2].mean().item()

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
    def _quat_axis(q: torch.Tensor, axis: int = 0) -> torch.Tensor:
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)
