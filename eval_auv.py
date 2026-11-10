import argparse

import matplotlib
matplotlib.use("Agg")

import torch

from env import Auv_blined, Auv_changeable
from policy import DQNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["q2", "q3"], required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-fuels", type=int, default=500)
    args = parser.parse_args()

    if args.env == "q2":
        env = Auv_changeable(args.max_fuels, "assets/eval_q2")
    else:
        env = Auv_blined(args.max_fuels, "assets/eval_q3")

    agent = DQNAgent(
        False,
        env.observation_space.shape[0],
        env.action_space.n,
        buffer_capacity=1000,
        batch_size=64,
    )
    agent.model.load_state_dict(torch.load(args.weight, map_location="cpu"))
    agent.model.eval()
    agent.epsilon = 0.0

    successes = 0
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
        success = int(total_reward > 0)
        successes += success
        print(
            f"episode={episode} reward={total_reward:.1f} "
            f"steps={env.current_time} success={bool(success)}"
        )

    env.render()
    print(f"success_rate={successes}/{args.episodes}")
    print(f"weight={args.weight}")


if __name__ == "__main__":
    main()
