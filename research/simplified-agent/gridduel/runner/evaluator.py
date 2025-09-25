from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .parallel_runner import MatchSpec, run_matches


@dataclass
class Opponent:
	name: str
	spec: str  # policy string


def simple_elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 16.0) -> Tuple[float, float]:
	"""Return updated Elo ratings after a single game.

	score_a is 1.0 for win, 0.5 for draw, 0.0 for loss.
	"""
	qa = 10 ** (rating_a / 400)
	qb = 10 ** (rating_b / 400)
	exp_a = qa / (qa + qb)
	exp_b = qb / (qa + qb)
	rating_a_new = rating_a + k * (score_a - exp_a)
	rating_b_new = rating_b + k * ((1.0 - score_a) - exp_b)
	return rating_a_new, rating_b_new


def evaluate_policy_against_pool(target_spec: str, opponents: List[Opponent], episodes: int = 50, parallel: int = 4, seed: int | None = None) -> Dict[str, Dict[str, float]]:
	"""Evaluate a target policy against multiple opponents, returning winrates and Elo.

	Returns dict mapping opponent name to {winrate, draws, losses, elo}.
	"""
	results: Dict[str, Dict[str, float]] = {}
	ratings: Dict[str, float] = {"target": 1000.0}
	for opp in opponents:
		ratings[opp.name] = 1000.0

	# Run matches per opponent
	for opp in opponents:
		spec = MatchSpec(policy0=target_spec, policy1=opp.spec, episodes=episodes, config={"seed": seed})
		res = run_matches([spec], n_workers=parallel, seed=seed)[0]
		wins = res["wins0"]
		losses = res["wins1"]
		draws = res["draws"]
		total = wins + losses + draws
		winrate = wins / total if total else 0.0
		# Elo update (single aggregate match): approximate score
		score = (wins + 0.5 * draws) / max(1, total)
		r_t, r_o = simple_elo_update(ratings["target"], ratings[opp.name], score)
		ratings["target"], ratings[opp.name] = r_t, r_o
		results[opp.name] = {"winrate": float(winrate), "draws": float(draws / max(1, total)), "losses": float(losses / max(1, total)), "elo": float(r_t)}

	return results