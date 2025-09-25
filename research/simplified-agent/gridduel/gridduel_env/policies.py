from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


class BasePolicy:
	"""Abstract policy interface."""

	def reset(self) -> None:
		return None

	def act(self, obs) -> int:
		raise NotImplementedError


class RandomPolicy(BasePolicy):
	def __init__(self, num_actions: int = 9, seed: Optional[int] = None) -> None:
		self.num_actions = int(num_actions)
		self.rng = np.random.default_rng(seed)

	def reset(self) -> None:
		return None

	def act(self, obs) -> int:
		return int(self.rng.integers(0, self.num_actions))


class RulePolicy(BasePolicy):
	"""Simple heuristic: if opponent in same row/col with clear line, fire. Else wander.

	Expects obs with channels [walls, bullets, self, opponent]. Works with either
	full grid or partial window.
	"""

	def __init__(self, num_actions: int = 9, seed: Optional[int] = None) -> None:
		self.num_actions = int(num_actions)
		self.rng = np.random.default_rng(seed)

	def reset(self) -> None:
		return None

	def act(self, obs) -> int:
		walls = obs[..., 0]
		self_mask = obs[..., 2]
		opp_mask = obs[..., 3]
		self_pos = np.argwhere(self_mask == 1)
		opp_pos = np.argwhere(opp_mask == 1)
		if self_pos.size == 0 or opp_pos.size == 0:
			return 0
		sy, sx = self_pos[0]
		oy, ox = opp_pos[0]
		# same row
		if sy == oy:
			lo, hi = sorted([sx, ox])
			if np.all(walls[sy, lo + 1 : hi] == 0):
				return 7 if ox < sx else 8
		# same col
		if sx == ox:
			lo, hi = sorted([sy, oy])
			if np.all(walls[lo + 1 : hi, sx] == 0):
				return 5 if oy < sy else 6
		# otherwise: small random walk with slight bias away from opponent
		dy = np.sign(sy - oy)
		dx = np.sign(sx - ox)
		candidates = [0, 1, 2, 3, 4]
		# prefer moving away
		preferred = []
		if dy < 0:
			preferred.append(2)
		elif dy > 0:
			preferred.append(1)
		if dx < 0:
			preferred.append(4)
		elif dx > 0:
			preferred.append(3)
		if preferred:
			if self.rng.random() < 0.7:
				return int(self.rng.choice(preferred))
		return int(self.rng.choice(candidates))


class HumanPolicy(BasePolicy):
	"""Keyboard/CLI policy. Works in notebooks or terminal via input()."""

	def __init__(self) -> None:
		self._mapping = {
			"x": 0,
			"w": 1,
			"s": 2,
			"a": 3,
			"d": 4,
			"i": 5,
			"k": 6,
			"j": 7,
			"l": 8,
		}

	def reset(self) -> None:
		return None

	def act(self, obs) -> int:
		print("Enter action: x=stay, wasd=move, ijkl=fire:", file=sys.stderr)
		try:
			cmd = input().strip().lower()
		except EOFError:
			cmd = "x"
		return int(self._mapping.get(cmd, 0))


class TorchModelPolicy(BasePolicy):
	"""PyTorch model policy. Expects a model with forward(obs_tensor)-> logits or actions.

	- The obs provided is HxWxC int8; we'll convert to float tensor CHW in [0,1].
	- If model outputs logits over 9 actions, pick argmax.
	- If model outputs an action already (scalar), cast to int.
	"""

	def __init__(self, checkpoint_path: str, device: Optional[str] = None) -> None:
		try:
			import torch  # type: ignore
		except Exception as e:  # noqa: BLE001
			raise RuntimeError("TorchModelPolicy requires torch. Install torch to use this policy.") from e
		self._torch = torch
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.model = torch.jit.load(checkpoint_path) if checkpoint_path.endswith(".pt") else torch.load(checkpoint_path, map_location=self.device)
		self.model.to(self.device)
		self.model.eval()

	def reset(self) -> None:
		return None

	def act(self, obs) -> int:
		import torch  # type: ignore

		arr = obs.astype(np.float32)
		arr = np.transpose(arr, (2, 0, 1)) / 1.0  # CHW; already 0/1
		t = torch.from_numpy(arr).unsqueeze(0).to(self.device)
		with torch.no_grad():
			out = self.model(t)
		if isinstance(out, (list, tuple)):
			out = out[0]
		if out.ndim == 2 and out.shape[1] >= 1:
			action = int(out.argmax(dim=1).item())
		else:
			action = int(out.item())
		return action


def make_policy_from_string(spec: str) -> BasePolicy:
	"""Factory parsing strings like 'random', 'rule', 'human', 'model:path.pt'."""
	if spec == "random":
		return RandomPolicy()
	if spec == "rule":
		return RulePolicy()
	if spec == "human":
		return HumanPolicy()
	if spec.startswith("model:"):
		path = spec.split(":", 1)[1]
		return TorchModelPolicy(path)
	raise ValueError(f"Unknown policy spec: {spec}")