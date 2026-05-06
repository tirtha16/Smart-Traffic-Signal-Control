# Smart Traffic Signal Control with Reinforcement Learning

An RL agent learns to control traffic-light timing at a 4-way intersection to minimize average queue length and waiting time, beating a fixed-timer baseline.

## Project structure

```
traffic_env.py          # custom 4-way intersection simulator (Gym-like API)
agents/
  fixed_timer.py        # baseline: switches phase every N seconds
  q_learning.py         # tabular Q-learning agent
  dqn.py                # DQN agent (PyTorch) with replay buffer + target net
train.py                # train DQN or Q-learning
evaluate.py             # compare all three agents on identical episodes
plot_training.py        # plot training curves
visualize.py            # pygame animation of cars + signal
```

## RL formulation

- **State (6-dim)**: normalized queue lengths for `[N, S, E, W]`, current phase, time-in-phase.
- **Actions (2)**: `0 = keep` current phase, `1 = switch` (subject to `min_green` and a yellow transition).
- **Reward**: `-(total_queue) - 0.1 * total_waiting_time` — penalizes both backlog and the time cars spend stuck.
- **Dynamics**: Poisson arrivals per lane (asymmetric by default — E/W is busier), 1 car/sec departure on green lanes, 3-sec yellow phase, 5-sec minimum green.

## Quick start

```bash
pip install -r requirements.txt

# Train DQN (≈1–2 minutes on CPU)
python train.py --agent dqn --episodes 300

# Train tabular Q-learning baseline
python train.py --agent qlearning --episodes 300

# Compare all three agents
python evaluate.py --episodes 20

# Plot training curves
python plot_training.py --agent dqn

# Watch the trained agent in action
python visualize.py --agent dqn
```
