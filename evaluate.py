"""Compares fixed-timer, tabular Q-learning, and DQN on identical traffic episodes."""

import argparse
import os

import numpy as np

from traffic_env import TrafficEnv


def run_episode(env, agent, seed):
    state = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        action = agent.act(state, greedy=True)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward, env.metrics()


def evaluate(n_episodes=20, steps_per_episode=500, ckpt_dir="checkpoints"):
    env = TrafficEnv(max_steps=steps_per_episode)

    from agents.fixed_timer import FixedTimerAgent
    agents = {"fixed_timer": FixedTimerAgent(period=20)}

    q_path = os.path.join(ckpt_dir, "qlearning.pkl")
    if os.path.exists(q_path):
        from agents.q_learning import QLearningAgent
        ql = QLearningAgent(env.n_actions)
        ql.load(q_path)
        ql.eps = 0.0
        agents["qlearning"] = ql

    dqn_path = os.path.join(ckpt_dir, "dqn.pt")
    if os.path.exists(dqn_path):
        from agents.dqn import DQNAgent
        dqn = DQNAgent(env.state_dim, env.n_actions)
        dqn.load(dqn_path)
        dqn.eps = 0.0
        agents["dqn"] = dqn

    results = {name: {"reward": [], "avg_queue": [], "avg_wait": [], "throughput": []}
               for name in agents}

    for ep in range(n_episodes):
        seed = 10_000 + ep
        for name, agent in agents.items():
            r, m = run_episode(env, agent, seed)
            results[name]["reward"].append(r)
            results[name]["avg_queue"].append(m["avg_queue"])
            results[name]["avg_wait"].append(m["avg_wait_per_step"])
            results[name]["throughput"].append(m["total_throughput"])

    print(f"\n=== Evaluation over {n_episodes} episodes ===\n")
    print(f"{'agent':<14} {'reward':>10} {'avg_queue':>10} {'avg_wait':>10} {'throughput':>12}")
    print("-" * 60)
    for name, data in results.items():
        print(
            f"{name:<14} {np.mean(data['reward']):>10.1f} "
            f"{np.mean(data['avg_queue']):>10.2f} "
            f"{np.mean(data['avg_wait']):>10.2f} "
            f"{np.mean(data['throughput']):>12.1f}"
        )

    if "fixed_timer" in results:
        base_q = np.mean(results["fixed_timer"]["avg_queue"])
        print("\nImprovement vs fixed_timer baseline (lower queue is better):")
        for name, data in results.items():
            if name == "fixed_timer":
                continue
            q = np.mean(data["avg_queue"])
            improvement = (base_q - q) / base_q * 100
            print(f"  {name:<14} {improvement:+.1f}% queue reduction")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    evaluate(args.episodes, args.steps)
