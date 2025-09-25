import argparse
import json
from typing import List, Tuple

from gridduel_env.env import GridDuelEnv
from gridduel_env.policies import make_policy_from_string, BasePolicy
from gridduel_env.utils import ReplayLogger, ReplayStep
from runner.parallel_runner import run_matches, MatchSpec


def run_single_match(policy0: BasePolicy, policy1: BasePolicy, episodes: int, seed: int | None, render: str | None, record: str | None) -> dict:
	stats = {"wins0": 0, "wins1": 0, "draws": 0, "episodes": 0, "avg_steps": 0.0}
	recorder = ReplayLogger() if record else None
	step_counts: List[int] = []

	for ep in range(episodes):
		env = GridDuelEnv(seed=None if seed is None else seed + ep)
		(policy0.reset(), policy1.reset())
		(obs0, obs1), _ = env.reset(seed=None if seed is None else seed + ep)
		if recorder:
			recorder.start_episode({"seed": None if seed is None else seed + ep})
			t = 0
		while True:
			a0 = int(policy0.act(obs0))
			a1 = int(policy1.act(obs1))
			(obs0, obs1), (r0, r1), done, info = env.step_both(a0, a1)
			if recorder:
				recorder.record_step(ReplayStep(step=t, positions=info["positions"], actions=(a0, a1), bullets=info["bullets"], rewards=(r0, r1)))
			if render == "ascii":
				print(env.render("ascii"))
				print()
			if done:
				if recorder:
					recorder.end_episode({"t": info.get("t", t)})
				step_counts.append(info.get("t", t))
				if r0 > r1:
					stats["wins0"] += 1
				elif r1 > r0:
					stats["wins1"] += 1
				else:
					stats["draws"] += 1
				break
				
			t += 1
		stats["episodes"] += 1

	stats["avg_steps"] = float(sum(step_counts) / max(1, len(step_counts)))
	if recorder and record:
		recorder.save_json(record)
	return stats


def aggregate_results(results: List[dict]) -> dict:
	total = {"wins0": 0, "wins1": 0, "draws": 0, "episodes": 0, "avg_steps": 0.0}
	steps = []
	for r in results:
		total["wins0"] += r.get("wins0", 0)
		total["wins1"] += r.get("wins1", 0)
		total["draws"] += r.get("draws", 0)
		total["episodes"] += r.get("episodes", 0)
		steps.append(r.get("avg_steps", 0.0))
	if steps:
		total["avg_steps"] = float(sum(steps) / len(steps))
	return total


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--player0", required=True)
	p.add_argument("--player1", required=True)
	p.add_argument("--episodes", type=int, default=1)
	p.add_argument("--parallel", type=int, default=1)
	p.add_argument("--render", choices=["ascii", "matplotlib"], default="ascii")
	p.add_argument("--seed", type=int, default=None)
	p.add_argument("--record", type=str, default=None)
	args = p.parse_args()

	if args.parallel and args.parallel > 1:
		# Split episodes into individual specs for parallelization
		specs = [
			MatchSpec(
				policy0=args.player0,
				policy1=args.player1,
				episodes=1,
				config={"seed": None if args.seed is None else args.seed + i},
			)
			for i in range(args.episodes)
		]
		results = run_matches(specs, n_workers=args.parallel, seed=args.seed)
		# run_matches returns list of dicts; aggregate
		agg = aggregate_results(results)
		print(json.dumps(agg, indent=2))
		return

	policy0 = make_policy_from_string(args.player0)
	policy1 = make_policy_from_string(args.player1)
	stats = run_single_match(policy0, policy1, episodes=args.episodes, seed=args.seed, render=args.render, record=args.record)
	print(json.dumps(stats, indent=2))


if __name__ == "__main__":
	main()