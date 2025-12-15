import numpy as np
import imageio.v2 as imageio
from air_hockey_mjenv import AirHockeyMujocoEnv
from QLearning import create_discrete_actions
from dqn_agent import DQNAgent
import argparse


def record_policy(env, agent, discrete_actions, episodes, filename, epsilon_eval=0.0):
    """
    Roll out a policy and save a GIF.

    - env must be created with render_mode="rgb_array"
    - agent must have select_action(state, epsilon)
    - discrete_actions is the same array used in training
    """
    frames = []

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            # Greedy (or slightly ε-greedy) policy
            action_idx = agent.select_action(state, epsilon_eval)
            action = discrete_actions[action_idx]

            next_state, reward, done, truncated, info = env.step(action)
            frame = env.render()
            frames.append(frame)

            state = next_state

    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved {filename}")


def train_dqn(
    num_episodes: int = 500,
    max_steps: int = 500,
    seed: int | None = 0,
    use_env_randomization: bool = False,
):
    if seed is not None:
        np.random.seed(seed)

    env = AirHockeyMujocoEnv(
        max_steps=max_steps,
        randomize_gravity=use_env_randomization,
        randomize_friction=use_env_randomization,
        randomize_restitution=use_env_randomization,
    )

    obs_dim = env.observation_space.shape[0]

    discrete_actions = create_discrete_actions()
    n_actions = discrete_actions.shape[0]

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        gamma=0.99,
        lr=5e-4,
        batch_size=64,
        buffer_capacity=100_000,
        min_buffer=5_000,  # start training after some random experience
        target_update_freq=1_000,
    )

    # ε schedule (over steps, not episodes)
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 500_000
    total_steps = 0

    def get_epsilon(t):
        frac = min(1.0, t / epsilon_decay_steps)
        return epsilon_start + frac * (epsilon_end - epsilon_start)

    episode_returns = []
    episode_lengths = []
    scored_flags = []
    conceded_flags = []

    best_return = -np.inf
    have_scoring_best = False

    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False

        total_reward = 0.0
        steps = 0
        scored_flag = False
        conceded_flag = False

        while not (done or truncated) and steps < max_steps:
            epsilon = get_epsilon(total_steps)
            action_idx = agent.select_action(state, epsilon)
            action_continuous = discrete_actions[action_idx]

            next_state, reward, done, truncated, info = env.step(action_continuous)

            if info.get("scored", False):
                scored_flag = True
            if info.get("conceded", False):
                conceded_flag = True

            agent.push_transition(
                state=state,
                action=action_idx,
                reward=reward,
                next_state=next_state,
                done=(done or truncated),
            )

            # one gradient update
            agent.train_step()

            state = next_state
            total_reward += reward
            steps += 1
            total_steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        scored_flags.append(1 if scored_flag else 0)
        conceded_flags.append(1 if conceded_flag else 0)

        if scored_flag:
            if (not have_scoring_best) or (total_reward > best_return):
                have_scoring_best = True
                best_return = total_reward
                agent.save("dqn_best_weights.pt")

        if (ep + 1) % 10 == 0:
            avg_ret = np.mean(episode_returns[-10:])
            avg_score = np.mean(scored_flags[-10:])
            print(
                f"Episode {ep + 1}/{num_episodes} | "
                f"AvgReturn(last10): {avg_ret:.2f} | "
                f"ScoreRate(last10): {avg_score:.2f} | "
                f"BestReturn: {best_return:.2f}"
            )

    env.close()

    episode_returns = np.array(episode_returns, dtype=np.float32)
    episode_lengths = np.array(episode_lengths, dtype=np.int32)
    scored_flags = np.array(scored_flags, dtype=np.int32)
    conceded_flags = np.array(conceded_flags, dtype=np.int32)

    print("\n=== DQN Training Finished ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean return: {episode_returns.mean():.3f}")
    print(f"Mean success rate: {scored_flags.mean():.3f}")
    print(f"Best single-episode return: {best_return:.3f}")

    np.save("dqn_episode_returns.npy", episode_returns)
    np.save("dqn_episode_scored_flags.npy", scored_flags)
    np.save("dqn_episode_conceded_flags.npy", conceded_flags)

    # Save last network as well
    agent.save("dqn_weights.pt")

    return (
        agent,
        episode_returns,
        episode_lengths,
        scored_flags,
        conceded_flags,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--randomize-env",
        action="store_true",
        help="Enable gravity/friction/restitution randomization in the environment",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=6000,
        help="Number of training episodes",
    )
    args = parser.parse_args()

    # Train DQN and get the final agent
    agent, ep_returns, ep_lengths, scored_flags, conceded_flags = train_dqn(
        num_episodes=args.episodes,
        max_steps=500,
        seed=0,
        use_env_randomization=args.randomize_env,
    )

    # Recreate discrete action set
    discrete_actions = create_discrete_actions()

    # Make an env for recording
    env = AirHockeyMujocoEnv(
        max_steps=500,
        render_mode="rgb_array",
        randomize_gravity=args.randomize_env,
        randomize_friction=args.randomize_env,
        randomize_restitution=args.randomize_env,
    )

    # Record the final greedy trained agent
    record_policy(
        env,
        agent,
        discrete_actions,
        episodes=5,
        filename=(
            "dqn_final_greedy_randomized.gif"
            if args.randomize_env
            else "dqn_final_greedy.gif"
        ),
        epsilon_eval=0.0,
    )
    env.close()
