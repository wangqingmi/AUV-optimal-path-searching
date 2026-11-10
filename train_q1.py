import argparse
import csv
import random
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from env import Auv_SimpleAction
from policy import DQNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--max-fuels", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="result/train_q1")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = Auv_SimpleAction(args.max_fuels, "assets/train_q1")
    agent = DQNAgent(
        False,
        env.observation_space.shape[0],
        env.action_space.n,
        args.buffer_capacity,
        args.batch_size,
    )

    log_path = args.output + "_log.csv"
    steps = 0
    start = time.time()

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "epsilon", "loss", "elapsed_s", "success"])

        for episode in range(1, args.episodes + 1):
            state = env.reset()
            done = False
            total_reward = 0.0
            loss_value = None

            while not done:
                steps += 1
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay()
                if loss is not None:
                    loss_value = float(loss.detach())
                state = next_state
                total_reward += reward

                if steps % args.target_update == 0:
                    agent.update_target_network()

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            elapsed = time.time() - start
            success = int(total_reward > 0)
            writer.writerow([
                episode,
                total_reward,
                env.current_time,
                agent.epsilon,
                loss_value,
                elapsed,
                success,
            ])
            f.flush()

            if episode == 1 or episode % 10 == 0:
                print(
                    f"episode={episode} reward={total_reward:.1f} "
                    f"steps={env.current_time} epsilon={agent.epsilon:.4f} "
                    f"elapsed={elapsed:.1f}s"
                )

            if args.save_every and episode % args.save_every == 0:
                agent.save_model(
                    f"{args.output}_ep{episode}.pth",
                    f"{args.output}_target_ep{episode}.pth",
                )

    agent.save_model(
        f"{args.output}_final.pth",
        f"{args.output}_target_final.pth",
    )
    env.render()
    env.visualize_metrics()
    print(f"model={args.output}_final.pth")
    print(f"log={log_path}")


if __name__ == "__main__":
    main()
