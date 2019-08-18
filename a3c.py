import argparse

from mathy.a3c.a3c_agent import A3CAgent

parser = argparse.ArgumentParser(description="Mathy a3c agent")
parser.add_argument(
    "--algorithm", default="a3c", type=str, help="Choose between 'a3c' and 'random'."
)
parser.add_argument(
    "--train", dest="train", action="store_true", help="Train our model."
)
parser.add_argument(
    "--lr", default=3e-4, help="Learning rate for the shared optimizer."
)
parser.add_argument(
    "--update-freq", default=20, type=int, help="How often to update the global model."
)
parser.add_argument(
    "--max-eps",
    default=10000,
    type=int,
    help="Global maximum number of episodes to run.",
)
parser.add_argument("--gamma", default=0.99, help="Discount factor of rewards.")
parser.add_argument(
    "--save-dir",
    default="training/a3c/",
    type=str,
    help="Directory in which you desire to save the model.",
)
args = parser.parse_args()

if __name__ == "__main__":
    import gym

    gym.envs.registration.register(id="mathy-v0", entry_point="gym_env:MathyGymEnv")
    gym.envs.registration.register(
        id="mathy-poly-v0", entry_point="gym_env:MathyGymPolyEnv"
    )
    gym.envs.registration.register(
        id="mathy-complex-v0", entry_point="gym_env:MathyGymComplexEnv"
    )
    gym.envs.registration.register(
        id="mathy-binomial-v0", entry_point="gym_env:MathyGymBinomialEnv"
    )
    agent = A3CAgent(args, "mathy-test-lstm")
    if args.train:
        agent.train()
    else:
        agent.play(True)
