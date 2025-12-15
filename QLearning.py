import numpy as np
from air_hockey_mjenv import AirHockeyMujocoEnv
import imageio.v2 as imageio


class LinearQAgent:
    """
    Linear function approximation for Q(s,a):

        Q(s, a) = w_a^T * phi(s)

    where phi(s) is a hand-crafted feature vector.
    """

    def __init__(
        self, n_features: int, n_actions: int, alpha: float = 5e-4, gamma: float = 0.99
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # Rough scales based on env's observation bounds:
        # x in [-0.75, 0.75], y in [-1.0, 1.0], v in ~[-10, 10]
        self.x_scale = 0.75
        self.y_scale = 1.0
        self.v_scale = 10.0

        # Max distance on table ~ sqrt(x_scale^2 + y_scale^2)
        self.dist_scale = np.sqrt(self.x_scale**2 + self.y_scale**2)

        # Weights: shape (n_actions, n_features)
        self.W = np.zeros((n_actions, n_features), dtype=np.float32)

    def featurize(self, state: np.ndarray) -> np.ndarray:
        """
        state: [x_p, y_p, vx_p, vy_p, x_pad, y_pad, vx_pad, vy_pad]

        Features (all roughly in [-1, 1] range + bias):
        - normalized puck state (4)
        - normalized paddle state (4)
        - normalized relative position (2)
        - normalized relative velocity (2)
        - normalized distance + squared distance (2)
        - bias term (1)

        Total: 15 features.
        """
        x_p, y_p, vx_p, vy_p, x_pad, y_pad, vx_pad, vy_pad = state

        # Normalize absolute positions / velocities
        x_p_n = x_p / self.x_scale
        y_p_n = y_p / self.y_scale
        vx_p_n = vx_p / self.v_scale
        vy_p_n = vy_p / self.v_scale

        x_pad_n = x_pad / self.x_scale
        y_pad_n = y_pad / self.y_scale
        vx_pad_n = vx_pad / self.v_scale
        vy_pad_n = vy_pad / self.v_scale

        # Relative features (also normalized)
        rel_x = x_p - x_pad
        rel_y = y_p - y_pad
        rel_vx = vx_p - vx_pad
        rel_vy = vy_p - vy_pad

        rel_x_n = rel_x / self.x_scale
        rel_y_n = rel_y / self.y_scale
        rel_vx_n = rel_vx / self.v_scale
        rel_vy_n = rel_vy / self.v_scale

        # Distance (normalized)
        dist = np.sqrt(rel_x**2 + rel_y**2)
        dist_n = dist / self.dist_scale

        feat = np.array(
            [
                x_p_n,
                y_p_n,
                vx_p_n,
                vy_p_n,
                x_pad_n,
                y_pad_n,
                vx_pad_n,
                vy_pad_n,
                rel_x_n,
                rel_y_n,
                rel_vx_n,
                rel_vy_n,
                dist_n,
                dist_n**2,
                1.0,  # bias term
            ],
            dtype=np.float32,
        )
        return feat

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute Q(s, :) for all actions given features phi(s).
        """
        return self.W @ phi  # (n_actions,)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        ε-greedy policy over Q(s,a).
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)

        phi = self.featurize(state)
        q_vals = self.q_values(phi)
        return int(np.argmax(q_vals))

    def update(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Q-learning update with linear function approximation:

            w_a <- w_a + alpha * (target - Q(s,a)) * grad_w Q(s,a)

        where grad_w Q(s,a) = phi(s) for the selected action.
        """
        phi = self.featurize(state)
        q_vals = self.q_values(phi)
        q_sa = q_vals[action_idx]

        if done:
            target = reward
        else:
            phi_next = self.featurize(next_state)
            q_next = self.q_values(phi_next)
            target = reward + self.gamma * np.max(q_next)

        td_error = target - q_sa

        # Only weights for the selected action are updated
        self.W[action_idx] += self.alpha * td_error * phi


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
    num_episodes: int = 1000,
    max_steps: int = 500,
    seed: int | None = 0,
):
    if seed is not None:
        np.random.seed(seed)

    env = AirHockeyMujocoEnv(max_steps=max_steps)
    discrete_actions = create_discrete_actions()
    n_actions = discrete_actions.shape[0]

    agent = LinearQAgent(
        n_features=15,
        n_actions=n_actions,
        alpha=5e-4,
        gamma=0.99,
    )

    # ε-greedy parameters
    epsilon_start = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    epsilon = epsilon_start

    episode_returns = []
    episode_lengths = []
    scored_flags = []
    conceded_flags = []
    epsilons = []

    # Track best agent (by reward, but prefer scoring episodes)
    best_return = -np.inf
    best_weights = None
    best_scored = False

    for ep in range(num_episodes):
        epsilons.append(epsilon)

        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        scored_flag = False
        conceded_flag = False

        while not (done or truncated):
            action_idx = agent.select_action(state, epsilon)
            action_continuous = discrete_actions[action_idx]

            next_state, reward, done, truncated, info = env.step(action_continuous)

            if info.get("scored", False):
                scored_flag = True
            if info.get("conceded", False):
                conceded_flag = True

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
        scored_flags.append(1 if scored_flag else 0)
        conceded_flags.append(1 if conceded_flag else 0)

        if scored_flag:
            # prefer any scoring episode over non-scoring ones
            if (not best_scored) or (total_reward > best_return):
                best_scored = True
                best_return = total_reward
                best_weights = agent.W.copy()
        else:
            # only update if we've never seen a scoring episode yet
            if (not best_scored) and (total_reward > best_return):
                best_return = total_reward
                best_weights = agent.W.copy()

        # ε decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Logging
        if (ep + 1) % 10 == 0:
            avg_ret = np.mean(episode_returns[-10:])
            avg_score = np.mean(scored_flags[-10:])
            print(
                f"Episode {ep + 1}/{num_episodes} | "
                f"AvgReturn(last10): {avg_ret:.4f} | "
                f"ScoreRate(last10): {avg_score:.2f}"
            )

    env.close()

    episode_returns = np.array(episode_returns, dtype=np.float32)
    episode_lengths = np.array(episode_lengths, dtype=np.int32)
    scored_flags = np.array(scored_flags, dtype=np.int32)
    conceded_flags = np.array(conceded_flags, dtype=np.int32)

    print("\n=== Q-learning Training Finished ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean return (all episodes): {episode_returns.mean():.4f}")
    print(f"Mean success rate (all episodes): {scored_flags.mean():.4f}")

    np.save("episode_returns.npy", episode_returns)
    np.save("episode_scored_flags.npy", scored_flags)
    np.save("episode_conceded_flags.npy", conceded_flags)
    np.save("trained_weights.npy", agent.W)
    if best_weights is not None:
        np.save("best_trained_weights.npy", best_weights)

    # Rebuild the best agent using the saved best weights
    best_agent = LinearQAgent(
        n_features=15,
        n_actions=agent.n_actions,
        alpha=agent.alpha,
        gamma=agent.gamma,
    )
    if best_weights is not None:
        best_agent.W = best_weights.copy()

    return (
        agent,
        best_agent,
        episode_returns,
        episode_lengths,
        scored_flags,
        conceded_flags,
        epsilons,
    )


if __name__ == "__main__":
    # Train
    agent, best_agent, ep_returns, ep_lengths, scored, conceded, eps = train_q_learning(
        num_episodes=10000,
        max_steps=500,
        seed=0,
    )
