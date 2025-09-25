"""GridDuel Environment

A simple two-player grid duel game environment.

Actions (per agent):
- 0: stay
- 1: move up
- 2: move down
- 3: move left
- 4: move right
- 5: fire up
- 6: fire down
- 7: fire left
- 8: fire right

Collision rule: swap is allowed when agents move into each other's cell in the same step.
If both agents attempt to move into the same third cell, both moves are cancelled (they stay).

Rewards:
- +1 for eliminating the opponent; -1 for being eliminated. 0 otherwise.
If both are eliminated in the same step, both receive 0 and the episode ends in a draw.

Observation:
- Full: shape (H, W, 4) int8: [walls, bullets, self, opponent]
- Partial (radius r): (2r+1, 2r+1, 4), zero-padded outside.

Use GridDuelEnvGym to train a single agent against a fixed opponent policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .utils import make_default_map, seed_numpy, render_ascii


Action = int
Position = Tuple[int, int]


@dataclass
class Bullet:
	pos: Position
	dir: Position  # (dy, dx)


_DIRS = {
	"up": (-1, 0),
	"down": (1, 0),
	"left": (0, -1),
	"right": (0, 1),
}

_ACTION_TO_MOVE = {
	0: (0, 0),
	1: _DIRS["up"],
	2: _DIRS["down"],
	3: _DIRS["left"],
	4: _DIRS["right"],
}

_ACTION_TO_FIRE = {
	5: _DIRS["up"],
	6: _DIRS["down"],
	7: _DIRS["left"],
	8: _DIRS["right"],
}


class GridDuelEnv(gym.Env):
	metadata = {"render_modes": ["ascii", "matplotlib"], "render_fps": 10}

	def __init__(
		self,
		H: int = 7,
		W: int = 7,
		max_steps: int = 200,
		walls: Optional[List[Position]] = None,
		partial_obs: Optional[int] = None,
		ammo: Optional[Dict[str, int]] = None,
		cooldown: Optional[int] = None,
		seed: Optional[int] = None,
	) -> None:
		self.height = int(H)
		self.width = int(W)
		self.max_steps = int(max_steps)
		self.partial_obs_radius = partial_obs
		self.ammo_cfg = ammo
		self.cooldown_cfg = cooldown
		self.rng = seed_numpy(seed)
		self._base_seed = int(self.rng.bit_generator._seed_seq.entropy)  # for reproducibility

		self.walls = walls if walls is not None else make_default_map(self.height, self.width)
		self.wall_mask = np.zeros((self.height, self.width), dtype=np.int8)
		for (y, x) in self.walls:
			self.wall_mask[y, x] = 1

		# Spaces
		obs_h = (2 * partial_obs + 1) if partial_obs is not None else self.height
		obs_w = (2 * partial_obs + 1) if partial_obs is not None else self.width
		self.observation_space = spaces.Box(low=0, high=1, shape=(obs_h, obs_w, 4), dtype=np.int8)
		self.action_space = spaces.Discrete(9)

		# State
		self.player_positions: List[Optional[Position]] = [None, None]
		self.player_alive = [True, True]
		self.player_cooldown = [0, 0]
		self.player_shots = [None, None]
		self.player_reload = [0, 0]
		self.bullets: List[Bullet] = []
		self.t = 0

		self._reset_ammo_state()

	def _reset_ammo_state(self) -> None:
		if self.ammo_cfg is None:
			self.player_shots = [None, None]
			self.player_reload = [0, 0]
		else:
			max_shots = int(self.ammo_cfg.get("max_shots", 3))
			reload_time = int(self.ammo_cfg.get("reload_time", 10))
			self.player_shots = [max_shots, max_shots]
			self.player_reload = [reload_time, reload_time]
			self._ammo_max_shots = max_shots
			self._ammo_reload_time = reload_time

	def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
		if seed is not None:
			self.rng = seed_numpy(seed)
			self._base_seed = int(self.rng.bit_generator._seed_seq.entropy)
		self.t = 0
		self.player_alive = [True, True]
		self.player_cooldown = [0, 0]
		self._reset_ammo_state()
		self.bullets = []

		# Sample starting positions not in walls and distinct
		free_cells = [(y, x) for y in range(self.height) for x in range(self.width) if self.wall_mask[y, x] == 0]
		self.rng.shuffle(free_cells)
		self.player_positions[0] = free_cells[0]
		self.player_positions[1] = free_cells[1]

		obs0 = self._encode_obs(0)
		obs1 = self._encode_obs(1)
		return (obs0, obs1), {}

	def step(self, action: int):
		raise NotImplementedError("Use GridDuelEnvGym or step_both for two-agent interaction")

	def step_both(self, a0: Action, a1: Action):
		if not all(self.player_alive):
			return (self._encode_obs(0), self._encode_obs(1)), (0.0, 0.0), True, {
				"positions": self.player_positions.copy(),
				"bullets": [(b.pos[0], b.pos[1]) for b in self.bullets],
			}

		self.t += 1

		# Movement intentions
		new_positions = [self.player_positions[0], self.player_positions[1]]
		actions = [a0, a1]
		moves = []
		for idx, a in enumerate(actions):
			if a in _ACTION_TO_MOVE:
				dy, dx = _ACTION_TO_MOVE[a]
				y, x = self.player_positions[idx]
				ny, nx = y + dy, x + dx
				if 0 <= ny < self.height and 0 <= nx < self.width and self.wall_mask[ny, nx] == 0:
					moves.append((idx, (ny, nx)))
				else:
					moves.append((idx, (y, x)))
			else:
				moves.append((idx, self.player_positions[idx]))

		# Resolve moves: detect conflicts
		proposed = {idx: pos for idx, pos in moves}
		# swap allowed
		if proposed[0] == self.player_positions[1] and proposed[1] == self.player_positions[0]:
			new_positions[0], new_positions[1] = proposed[0], proposed[1]
		else:
			# If both propose same cell (not a swap), cancel both
			if proposed[0] == proposed[1] and proposed[0] != self.player_positions[0] and proposed[1] != self.player_positions[1]:
				new_positions = [self.player_positions[0], self.player_positions[1]]
			else:
				for i in (0, 1):
					new_positions[i] = proposed[i]

		self.player_positions = new_positions

		# Shooting: respect cooldown and ammo
		for i, a in enumerate(actions):
			if a in _ACTION_TO_FIRE and self.player_alive[i]:
				if self.player_cooldown[i] > 0:
					pass
				else:
					if self.ammo_cfg is None or (self.player_shots[i] is not None and self.player_shots[i] > 0):
						dy, dx = _ACTION_TO_FIRE[a]
						y, x = self.player_positions[i]
						ny, nx = y + dy, x + dx
						if 0 <= ny < self.height and 0 <= nx < self.width and self.wall_mask[ny, nx] == 0:
							self.bullets.append(Bullet(pos=(ny, nx), dir=(dy, dx)))
							if self.cooldown_cfg:
								self.player_cooldown[i] = int(self.cooldown_cfg)
							if self.ammo_cfg is not None and self.player_shots[i] is not None:
								self.player_shots[i] -= 1

		# Update cooldowns and ammo reload timers
		for i in (0, 1):
			if self.player_cooldown[i] > 0:
				self.player_cooldown[i] -= 1
			if self.ammo_cfg is not None and self.player_shots[i] is not None:
				# simple reload: increment one shot every reload_time steps
				if self.t % getattr(self, "_ammo_reload_time", 10) == 0 and self.player_shots[i] < getattr(self, "_ammo_max_shots", 3):
					self.player_shots[i] += 1

		# Move bullets and resolve hits
		new_bullets: List[Bullet] = []
		for b in self.bullets:
			ny, nx = b.pos[0] + b.dir[0], b.pos[1] + b.dir[1]
			if not (0 <= ny < self.height and 0 <= nx < self.width):
				continue
			if self.wall_mask[ny, nx] == 1:
				continue
			# check hit
			for i in (0, 1):
				if self.player_alive[i] and (ny, nx) == self.player_positions[i]:
					self.player_alive[i] = False
					break
			else:
				new_bullets.append(Bullet(pos=(ny, nx), dir=b.dir))
		self.bullets = new_bullets

		# Rewards and termination
		rewards = [0.0, 0.0]
		if not self.player_alive[0] and self.player_alive[1]:
			rewards = [-1.0, 1.0]
		elif not self.player_alive[1] and self.player_alive[0]:
			rewards = [1.0, -1.0]
		elif not self.player_alive[0] and not self.player_alive[1]:
			rewards = [0.0, 0.0]

		done = self.t >= self.max_steps or (not self.player_alive[0]) or (not self.player_alive[1])

		obs0 = self._encode_obs(0)
		obs1 = self._encode_obs(1)
		info = {
			"positions": self.player_positions.copy(),
			"bullets": [(b.pos[0], b.pos[1]) for b in self.bullets],
			"alive": self.player_alive.copy(),
			"t": self.t,
		}
		return (obs0, obs1), (float(rewards[0]), float(rewards[1])), bool(done), info

	def _encode_obs(self, agent_idx: int) -> np.ndarray:
		self_mask = np.zeros((self.height, self.width), dtype=np.int8)
		opp_mask = np.zeros((self.height, self.width), dtype=np.int8)
		if self.player_positions[agent_idx] is not None and self.player_alive[agent_idx]:
			y, x = self.player_positions[agent_idx]
			self_mask[y, x] = 1
		other = 1 - agent_idx
		if self.player_positions[other] is not None and self.player_alive[other]:
			y, x = self.player_positions[other]
			opp_mask[y, x] = 1
		bullet_mask = np.zeros((self.height, self.width), dtype=np.int8)
		for b in self.bullets:
			bullet_mask[b.pos[0], b.pos[1]] = 1
		full = np.stack([self.wall_mask, bullet_mask, self_mask, opp_mask], axis=-1)
		if self.partial_obs_radius is None:
			return full
		# crop around agent
		r = int(self.partial_obs_radius)
		ah, aw = 2 * r + 1, 2 * r + 1
		pad = ((r, r), (r, r), (0, 0))
		padded = np.pad(full, pad, mode="constant")
		agent_pos = self.player_positions[agent_idx]
		if agent_pos is None:
			return np.zeros((ah, aw, 4), dtype=np.int8)
		y, x = agent_pos
		y0, x0 = y + r, x + r
		crop = padded[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1, :]
		return crop

	def render(self, mode: str = "ascii") -> Any:
		if mode == "ascii":
			return render_ascii(
				self.height,
				self.width,
				self.walls,
				self.player_positions,
				[(b.pos[0], b.pos[1]) for b in self.bullets],
			)
		elif mode == "matplotlib":
			import matplotlib.pyplot as plt

			fig, ax = plt.subplots(figsize=(self.width / 2, self.height / 2))
			ax.set_xlim(-0.5, self.width - 0.5)
			ax.set_ylim(-0.5, self.height - 0.5)
			ax.invert_yaxis()
			ax.set_xticks(range(self.width))
			ax.set_yticks(range(self.height))
			ax.grid(True, which="both")

			# draw walls
			for (y, x) in self.walls:
				ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="black"))
			# players
			colors = ["tab:blue", "tab:red"]
			for i, pos in enumerate(self.player_positions):
				if pos is not None and self.player_alive[i]:
					y, x = pos
					ax.add_patch(plt.Circle((x, y), 0.3, color=colors[i]))
			# bullets
			for b in self.bullets:
				y, x = b.pos
				ax.add_patch(plt.Circle((x, y), 0.1, color="gold"))
			ax.set_title(f"t={self.t}")
			fig.tight_layout()
			return fig
		else:
			raise ValueError(f"Unknown render mode: {mode}")


class GridDuelEnvGym(gym.Env):
	"""Single-agent Gym wrapper to play against a fixed opponent policy.

	The wrapper exposes the controlled player's observation and reward. On each
	step, it queries the opponent policy for an action using the opponent's view.
	"""

	def __init__(self, opponent_policy, player_index: int = 0, **env_kwargs) -> None:
		self.player_index = int(player_index)
		self.opponent_index = 1 - self.player_index
		self.core = GridDuelEnv(**env_kwargs)
		self.opponent = opponent_policy

		self.action_space = self.core.action_space
		self.observation_space = self.core.observation_space

		self._last_joint_obs = None

	def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
		joint_obs, info = self.core.reset(seed=seed)
		self.opponent.reset()
		self._last_joint_obs = joint_obs
		return joint_obs[self.player_index], info

	def step(self, action: int):
		# Opponent acts on their observation
		opp_obs = self._last_joint_obs[self.opponent_index]
		opp_action = int(self.opponent.act(opp_obs))
		joint_obs, rewards, done, info = self.core.step_both(
			int(action) if action is not None else 0, opp_action
		)
		self._last_joint_obs = joint_obs
		obs = joint_obs[self.player_index]
		reward = float(rewards[self.player_index])
		terminated = bool(done)
		truncated = False
		return obs, reward, terminated, truncated, info