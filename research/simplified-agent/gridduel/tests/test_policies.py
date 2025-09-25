from unittest.mock import patch

from gridduel_env.policies import RandomPolicy, RulePolicy, HumanPolicy


def test_random_policy_range():
	p = RandomPolicy()
	for _ in range(10):
		a = p.act(None)
		assert 0 <= a < 9


def test_rule_policy_basic():
	p = RulePolicy()
	# trivial obs with self and opp separated; expect valid action
	import numpy as np
	obs = np.zeros((5, 5, 4), dtype=np.int8)
	obs[..., 0] = 0  # walls
	obs[2, 2, 2] = 1  # self
	obs[2, 4, 3] = 1  # opp same row
	a = p.act(obs)
	assert 0 <= a < 9


def test_human_policy_mocked():
	p = HumanPolicy()
	with patch("builtins.input", return_value="w"):
		a = p.act(None)
		assert a == 1