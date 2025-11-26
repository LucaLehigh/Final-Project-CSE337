import numpy as np
from air_hockey_env import AirHockeyEnv


class LinearQAgent:
    """
    Feature-based Q-learning with linear function approximation.

    Q(s, a) = w_a^T * phi(s)
    where phi(s) is a feature vector of the state.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        alpha: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Weights: shape (n_actions, n_features)
        self.W = np.zeros((n_actions, n_features), dtype=np.float32)

    def featurize(self, state: np.ndarray) -> np.ndarray:
        """
        Simple hand-crafted features.

        state: [x_p, y_p, vx_p, vy_p, x_pad, y_pad, vx_pad, vy_pad]

        Features:
        - raw state
        - relative position (puck - paddle)
        - relative velocity (puck - paddle)
        """
        x_p, y_p, vx_p, vy_p, x_pad, y_pad, vx_pad, vy_pad = state

        rel_x = x_p - x_pad
        rel_y = y_p - y_pad
        rel_vx = vx_p - vx_pad
        rel_vy = vy_p - vy_pad

        # Optionally scale some terms
        feat = np.array(
            [
                x_p,
                y_p,
                vx_p,
                vy_p,
                x_pad,
                y_pad,
                vx_pad,
                vy_pad,
                rel_x,
                rel_y,
                rel_vx,
                rel_vy,
            ],
            dtype=np.float32,
        )

        return feat

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute Q(s, :) for all actions given features phi(s).
        """
        return self.W @ phi  # shape (n_actions,)

    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy over Q(s,a).
        """
        phi = self.featurize(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = self.q_values(phi)
        return int(np.argmax(q_vals))

    def update(self, state, action_idx, reward, next_state, done):
        """
        Q-learning update with linear function approximation.
        """
        phi = self.featurize(state)  # shape (n_features,)
        q_vals = self.q_values(phi)  # shape (n_actions,)
        q_sa = q_vals[action_idx]

        if done:
            target = reward
        else:
            phi_next = self.featurize(next_state)
            q_next = self.q_values(phi_next)
            target = reward + self.gamma * np.max(q_next)

        td_error = target - q_sa

        # Gradient of Q wrt W[action] is just phi
        self.W[action_idx] += self.alpha * td_error * phi

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def create_discrete_actions():
    """
    Map a small discrete action set to continuous accelerations in env action space.

    Actions:
        0: stay
        1: left
        2: right
        3: up
        4: down
    """
    ACTION_MAG = 1.0  # max magnitude in each dimension, consistent with env's [-1,1]

    actions = np.array(
        [
            [0.0, 0.0],  # stay
            [-ACTION_MAG, 0.0],  # left
            [ACTION_MAG, 0.0],  # right
            [0.0, ACTION_MAG],  # up (positive y)
            [0.0, -ACTION_MAG],  # down (negative y)
        ],
        dtype=np.float32,
    )
    return actions


def train_q_learning(
    num_episodes: int = 500,
    max_steps: int = 500,
    seed: int | None = 0,
):
    if seed is not None:
        np.random.seed(seed)

    env = AirHockeyEnv(max_steps=max_steps)
    discrete_actions = create_discrete_actions()
    n_actions = discrete_actions.shape[0]

    # Features: see featurize() => 12-dimensional
    agent = LinearQAgent(
        n_features=12,
        n_actions=n_actions,
        alpha=5e-4,
        gamma=0.99,
        epsilon=0.3,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            # Select discrete action index using ε-greedy policy
            action_idx = agent.select_action(state)
            action_continuous = discrete_actions[action_idx]

            next_state, reward, done, truncated, info = env.step(action_continuous)

            agent.update(
                state=state,
                action_idx=action_idx,
                reward=reward,
                next_state=next_state,
                done=(done or truncated),
            )

            state = next_state
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

        if (ep + 1) % 10 == 0:
            avg_ret = np.mean(episode_returns[-10:])
            print(
                f"Episode {ep + 1}/{num_episodes} "
                f"- AvgReturn(last10): {avg_ret:.4f}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    episode_returns = np.array(episode_returns, dtype=np.float32)
    episode_lengths = np.array(episode_lengths, dtype=np.int32)

    print("\n=== Q-learning Training Finished ===")
    print(f"Episodes: {num_episodes}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Mean return (all episodes): {episode_returns.mean():.4f}")
    print(f"Mean length (all episodes): {episode_lengths.mean():.2f}")

    return agent, episode_returns, episode_lengths


if __name__ == "__main__":
    train_q_learning(
        num_episodes=500,
        max_steps=500,
        seed=0,
    )
