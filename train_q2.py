import argparse
import csv
import random
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from env import Auv_changeable
from policy import DQNAgent


def evaluate(env, agent, episodes):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    successes = 0
    total_steps = 0
    total_reward = 0.0

    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            episode_reward += reward
        successes += int(episode_reward > 0)
        total_steps += env.current_time
        total_reward += episode_reward

    agent.epsilon = old_epsilon
    return successes, total_steps / episodes, total_reward / episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-fuels", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="result/train_q2")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = Auv_changeable(args.max_fuels, "assets/train_q2")
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
    best_success = -1
    best_steps = float("inf")

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

            if args.eval_every and episode % args.eval_every == 0:
                eval_success, eval_steps, eval_reward = evaluate(env, agent, args.eval_episodes)
                print(
                    f"eval episode={episode} success={eval_success}/{args.eval_episodes} "
                    f"mean_steps={eval_steps:.1f} mean_reward={eval_reward:.1f}"
                )
                if eval_success > best_success or (
                    eval_success == best_success and eval_steps < best_steps
                ):
                    best_success = eval_success
                    best_steps = eval_steps
                    agent.save_model(
                        f"{args.output}_best.pth",
                        f"{args.output}_target_best.pth",
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
    print(f"best_model={args.output}_best.pth")
    print(f"log={log_path}")


if __name__ == "__main__":
    main()
