# Utility helpers for GridDuel
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def make_default_map(height: int, width: int) -> List[Tuple[int, int]]:
	"""Create a simple symmetric map with a few walls.

	Places a border of walls and a small plus-shaped obstacle in the center.
	Returns a list of (y, x) tuples for wall coordinates.
	"""
	walls = []
	for x in range(width):
		walls.append((0, x))
		walls.append((height - 1, x))
	for y in range(height):
		walls.append((y, 0))
		walls.append((y, width - 1))

	cy, cx = height // 2, width // 2
	for dy in (-1, 0, 1):
		walls.append((cy + dy, cx))
	for dx in (-1, 0, 1):
		walls.append((cy, cx + dx))

	# Remove duplicates and out-of-bounds
	walls = [(y, x) for (y, x) in set(walls) if 0 <= y < height and 0 <= x < width]
	return walls


def seed_numpy(seed: Optional[int]) -> np.random.Generator:
	"""Return a numpy RNG with the given seed (or a random one if None)."""
	if seed is None:
		seed = np.random.SeedSequence().entropy
	return np.random.default_rng(int(seed))


def render_ascii(
	height: int,
	width: int,
	walls: List[Tuple[int, int]],
	player_positions: List[Optional[Tuple[int, int]]],
	bullets: List[Tuple[int, int]],
) -> str:
	"""Render an ASCII representation of the grid.

	Legend:
	- '#' wall
	- '.' empty
	- '0' player 0
	- '1' player 1
	- '*' bullet
	"""
	grid = [["." for _ in range(width)] for _ in range(height)]
	for (y, x) in walls:
		grid[y][x] = "#"
	for (y, x) in bullets:
		if 0 <= y < height and 0 <= x < width:
			grid[y][x] = "*"
	for idx, pos in enumerate(player_positions):
		if pos is not None:
			y, x = pos
			if 0 <= y < height and 0 <= x < width:
				grid[y][x] = str(idx)
	return "\n".join("".join(row) for row in grid)


@dataclass
class ReplayStep:
	step: int
	positions: List[Optional[Tuple[int, int]]]
	actions: Tuple[int, int]
	bullets: List[Tuple[int, int]]
	rewards: Tuple[float, float]


class ReplayLogger:
	"""Simple replay logger that accumulates steps and writes to JSON."""

	def __init__(self) -> None:
		self.episodes: List[dict] = []
		self._current: Optional[dict] = None

	def start_episode(self, config: dict) -> None:
		self._current = {"config": config, "steps": []}

	def record_step(self, entry: ReplayStep) -> None:
		assert self._current is not None, "Call start_episode first"
		self._current["steps"].append(
			{
				"step": entry.step,
				"positions": entry.positions,
				"actions": list(entry.actions),
				"bullets": entry.bullets,
				"rewards": list(entry.rewards),
			}
		)

	def end_episode(self, summary: dict) -> None:
		assert self._current is not None, "Call start_episode first"
		self._current["summary"] = summary
		self.episodes.append(self._current)
		self._current = None

	def save_json(self, path: str) -> None:
		with open(path, "w", encoding="utf-8") as f:
			json.dump({"episodes": self.episodes}, f, indent=2)