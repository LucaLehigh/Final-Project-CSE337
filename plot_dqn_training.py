import numpy as np
import matplotlib.pyplot as plt

# ===== Load saved DQN training data =====
returns = np.load("dqn_episode_returns.npy")  # shape (N,)
scores = np.load("dqn_episode_scored_flags.npy")  # 0/1 per episode
conceded = np.load("dqn_episode_conceded_flags.npy")  # 0/1 per episode

N = len(returns)
print(f"Loaded {N} episodes")


# ===== Helper: moving average =====
def moving_average(x, window: int):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


window = 20  # smoothing window size

returns_ma = moving_average(returns, window)
scores_ma = moving_average(scores, window)
goal_diff = scores - conceded  # +1 win, -1 loss, 0 tie
goal_diff_ma = moving_average(goal_diff, window)

ma_x = np.arange(window - 1, N)

# ===========================
# 1) Returns per episode
# ===========================
plt.figure(figsize=(8, 4))
plt.plot(returns, alpha=0.3, label="Return per episode")
plt.plot(ma_x, returns_ma, linewidth=2, label=f"{window}-episode moving avg")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DQN Training: Episode Return")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("dqn_training_returns.png", dpi=200)

# ===========================
# 2) Scoring frequency
# ===========================
plt.figure(figsize=(8, 4))
plt.plot(scores, alpha=0.3, label="Scored (0/1) per episode")
plt.plot(ma_x, scores_ma, linewidth=2, label=f"{window}-episode moving avg")
plt.xlabel("Episode")
plt.ylabel("Scoring indicator / rate")
plt.title("DQN Training: Scoring Frequency")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("dqn_training_success.png", dpi=200)

# ===========================
# 3) Goal differential (score - concede)
# ===========================
plt.figure(figsize=(8, 4))
plt.plot(goal_diff, alpha=0.3, label="Score - Concede per episode")
plt.plot(ma_x, goal_diff_ma, linewidth=2, label=f"{window}-episode moving avg")
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Goal differential")
plt.title("DQN Training: Goal Differential")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("dqn_training_goal_diff.png", dpi=200)

print(
    "Saved dqn_training_returns.png, dqn_training_success.png, dqn_training_goal_diff.png"
)
