import argparse

import matplotlib
matplotlib.use("Agg")

import torch
from env import Auv_SimpleAction
from policy import DQNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="result/Simple-5000-1000--model.pth")
    args = parser.parse_args()

    env = Auv_SimpleAction(max_fuels=500, image_save_path="assets/eval_q1")

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

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
        total_reward += reward

    env.render()
    print(f"total_reward={total_reward}")
    print(f"steps={env.current_time}")
    print(f"path_points={len(env.trajectory)}")
    print(f"weight={args.weight}")
    print("saved=assets/eval_q1.png")


if __name__ == "__main__":
    main()
