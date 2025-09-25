from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env import GridDuelEnv, GridDuelEnvGym


class PartialObsWrapper(gym.Wrapper):
	"""Forces partial observation radius on the underlying GridDuelEnv."""

	def __init__(self, env: GridDuelEnv, radius: int) -> None:
		assert isinstance(env, GridDuelEnv)
		super().__init__(env)
		self.env: GridDuelEnv
		self.env.partial_obs_radius = int(radius)
		obs_h = 2 * radius + 1
		obs_w = 2 * radius + 1
		self.observation_space = spaces.Box(low=0, high=1, shape=(obs_h, obs_w, 4), dtype=np.int8)


class GymWrapperForSingleAgent(GridDuelEnvGym):
	"""Alias for clarity per spec."""


class VecEnvAdapter:
	"""Minimal vectorized env adapter for a list of GridDuelEnvGym-like envs."""

	def __init__(self, env_fns: Iterable[Callable[[], gym.Env]]):
		self.envs = [fn() for fn in env_fns]
		self.num_envs = len(self.envs)
		self.single_observation_space = self.envs[0].observation_space
		self.single_action_space = self.envs[0].action_space

	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
		obs_list = []
		infos = []
		for i, env in enumerate(self.envs):
			obs, info = env.reset(seed=None if seed is None else int(seed) + i)
			obs_list.append(obs)
			infos.append(info)
		return np.stack(obs_list, axis=0), infos

	def step(self, actions: List[int]):
		obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
		for env, a in zip(self.envs, actions):
			obs, r, term, trunc, info = env.step(a)
			obs_list.append(obs)
			rew_list.append(r)
			term_list.append(term)
			trunc_list.append(trunc)
			info_list.append(info)
		return (
			np.stack(obs_list, axis=0),
			np.array(rew_list, dtype=np.float32),
			np.array(term_list, dtype=bool),
			np.array(trunc_list, dtype=bool),
			info_list,
		)