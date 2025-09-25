from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

from gridduel_env.env import GridDuelEnv
from gridduel_env.policies import make_policy_from_string


@dataclass
class MatchSpec:
	policy0: str
	policy1: str
	episodes: int = 1
	config: Optional[Dict[str, Any]] = None


@dataclass
class MatchResult:
	wins0: int
	wins1: int
	draws: int
	avg_steps: float
	episodes: int
	meta: Dict[str, Any]


def _worker_run(spec: MatchSpec, base_seed: Optional[int]) -> MatchResult:
	seed = None
	if base_seed is not None:
		# incorporate PID to vary per-process seeds
		seed = int(base_seed + os.getpid())
	policy0 = make_policy_from_string(spec.policy0)
	policy1 = make_policy_from_string(spec.policy1)
	wins0 = wins1 = draws = 0
	steps: List[int] = []
	for ep in range(spec.episodes):
		env = GridDuelEnv(seed=None if seed is None else seed + ep)
		policy0.reset(); policy1.reset()
		(obs0, obs1), _ = env.reset(seed=None if seed is None else seed + ep)
		while True:
			a0 = int(policy0.act(obs0))
			a1 = int(policy1.act(obs1))
			(obs0, obs1), (r0, r1), done, info = env.step_both(a0, a1)
			if done:
				steps.append(info.get("t", 0))
				if r0 > r1:
					wins0 += 1
				elif r1 > r0:
					wins1 += 1
				else:
					draws += 1
				break
	avg_steps = float(sum(steps) / max(1, len(steps)))
	return MatchResult(wins0=wins0, wins1=wins1, draws=draws, avg_steps=avg_steps, episodes=spec.episodes, meta=spec.config or {})


def run_matches(match_specs: List[MatchSpec], n_workers: int = 4, seed: Optional[int] = None, callback: Optional[Callable[[MatchResult], None]] = None) -> List[Dict[str, Any]]:
	"""Run many matches in parallel across worker processes.

	Returns a list of dict results. If callback is given, it is called per result.
	"""
	results: List[Dict[str, Any]] = []
	with ProcessPoolExecutor(max_workers=max(1, n_workers)) as ex:
		futures = [ex.submit(_worker_run, spec, seed) for spec in match_specs]
		for fut in as_completed(futures):
			res = fut.result()
			res_d = asdict(res)
			results.append(res_d)
			if callback is not None:
				callback(res)
	return results