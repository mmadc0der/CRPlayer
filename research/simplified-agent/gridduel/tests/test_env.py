import numpy as np

from gridduel_env.env import GridDuelEnv


def test_reset_shapes_full_obs():
	env = GridDuelEnv(H=7, W=7, partial_obs=None)
	(obs0, obs1), _ = env.reset(seed=123)
	assert obs0.shape == (7, 7, 4)
	assert obs1.shape == (7, 7, 4)


def test_step_both_determinism():
	env1 = GridDuelEnv(H=5, W=5)
	env2 = GridDuelEnv(H=5, W=5)
	(obs0a, obs1a), _ = env1.reset(seed=42)
	(obs0b, obs1b), _ = env2.reset(seed=42)
	for _ in range(5):
		(o1, o2), (r1, r2), d1, info1 = env1.step_both(1, 4)
		(o3, o4), (r3, r4), d2, info2 = env2.step_both(1, 4)
		assert np.array_equal(o1, o3)
		assert np.array_equal(o2, o4)
		assert r1 == r3 and r2 == r4 and d1 == d2


def test_bullet_hit_logic():
	env = GridDuelEnv(H=5, W=5, walls=[])
	(obs0, obs1), _ = env.reset(seed=7)
	# Force positions for test: place players aligned vertically
	env.player_positions = [(2, 2), (0, 2)]
	env.player_alive = [True, True]
	# Player 0 fires up
	(obs0, obs1), (r0, r1), done, info = env.step_both(5, 0)
	# Bullet should move next step and hit player1
	for _ in range(2):
		(obs0, obs1), (r0, r1), done, info = env.step_both(0, 0)
	assert (not env.player_alive[1]) or done