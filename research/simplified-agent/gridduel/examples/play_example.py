from gridduel_env.env import GridDuelEnv
from gridduel_env.policies import RandomPolicy, RulePolicy


def main():
	env = GridDuelEnv()
	p0 = RandomPolicy()
	p1 = RulePolicy()
	(obs0, obs1), _ = env.reset()
	while True:
		a0 = p0.act(obs0)
		a1 = p1.act(obs1)
		(obs0, obs1), (r0, r1), done, info = env.step_both(a0, a1)
		print(env.render("ascii"))
		print()
		if done:
			print("Rewards:", r0, r1)
			break


if __name__ == "__main__":
	main()