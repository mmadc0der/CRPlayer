from time import perf_counter

import numpy as np

from gridduel_env.env import GridDuelEnvGym
from gridduel_env.policies import RandomPolicy


def main():
	# Train an agent against RandomPolicy using a trivial policy gradient-like loop (stub)
	opponent = RandomPolicy()
	env = GridDuelEnvGym(opponent_policy=opponent, player_index=0)

	episodes = 10
	total_reward = 0.0
	for ep in range(episodes):
		obs, _ = env.reset()
		done = False
		episode_reward = 0.0
		while not done:
			action = np.random.randint(env.action_space.n)
			obs, reward, terminated, truncated, info = env.step(int(action))
			episode_reward += reward
			done = terminated or truncated
		print(f"Episode {ep}: reward={episode_reward:.2f}")
		total_reward += episode_reward
	print(f"Avg reward: {total_reward / episodes:.2f}")


if __name__ == "__main__":
	main()