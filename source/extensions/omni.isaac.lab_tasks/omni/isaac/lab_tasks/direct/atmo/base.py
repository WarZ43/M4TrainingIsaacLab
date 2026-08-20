from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .task_utils import RewardMixer, stage_value


class BaseTask(ABC):
    """The task half of an M4 environment."""

    def __init__(
        self,
        env,
        cfg,
        reward_keys: tuple[str, ...],
        episode_log_keys: tuple[str, ...] = (),
    ):
        self.env = env
        self.cfg = cfg
        self.reward_mixer = RewardMixer(reward_keys)
        self._stage_value_cache: dict[tuple[int, str, int], object] = {}
        self._logged_episode_reward_keys = tuple(dict.fromkeys((*reward_keys, *episode_log_keys)))
        self._episode_reward_sums = {
            key: torch.zeros(env.num_envs, device=env.device)
            for key in self._logged_episode_reward_keys
        }
        self._pending_episode_reward_totals = {
            key: torch.zeros((), device=env.device)
            for key in self._logged_episode_reward_keys
        }
        self._pending_episode_reward_counts = {key: 0 for key in self._logged_episode_reward_keys}

    def stage_value(self, values, name: str):
        stage = int(self.env.curriculum_stage_for_value(name))
        key = (id(values), name, stage)
        if key not in self._stage_value_cache:
            self._stage_value_cache[key] = stage_value(self.env, values, name)
        return self._stage_value_cache[key]

    def log_rewards(
        self,
        rewards: dict[str, torch.Tensor],
        done: torch.Tensor,
    ) -> None:
        env = self.env
        env.extras.pop("log", None)
        for key in self._logged_episode_reward_keys:
            self._episode_reward_sums[key].add_(rewards[key])

        if not torch.any(done):
            return

        for key, value in self._episode_reward_sums.items():
            completed = self.episode_reward_done_mask(key, done)
            completed_count = int(completed.sum().item())
            if completed_count > 0:
                self._pending_episode_reward_totals[key] += value[completed].sum()
                self._pending_episode_reward_counts[key] += completed_count
            value[done] = 0.0

        log = env.extras.setdefault("log", {})
        for key, total in self._pending_episode_reward_totals.items():
            count = self._pending_episode_reward_counts[key]
            log[f"Episode_Reward/{key}"] = total / float(count) if count > 0 else total
            total.zero_()
            self._pending_episode_reward_counts[key] = 0

    def episode_reward_done_mask(self, key: str, done: torch.Tensor) -> torch.Tensor:
        return done

    @abstractmethod
    def reset_initial_state(self, env_ids, randomized: bool): ...

    @abstractmethod
    def get_rewards(self): ...

    @abstractmethod
    def get_dones(self): ...
