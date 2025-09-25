from runner.parallel_runner import run_matches, MatchSpec


def test_parallel_runner_smoke():
	spec = MatchSpec(policy0="random", policy1="random", episodes=4, config={"seed": 123})
	res_list = run_matches([spec], n_workers=2, seed=123)
	assert isinstance(res_list, list) and len(res_list) == 1
	res = res_list[0]
	assert set(["wins0", "wins1", "draws", "avg_steps", "episodes", "meta"]).issubset(res.keys())
	assert res["episodes"] == 4