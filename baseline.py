import numpy as np
from air_hockey_env import AirHockeyEnv


def run_random_baseline(
    num_episodes: int = 100,
    max_steps: int = 500,
    seed: int | None = 0,
):
    """
    Run a random policy on AirHockeyEnv to produce a simple baseline.

    Logs:
        - episode return
        - episode length
    """
    if seed is not None:
        np.random.seed(seed)

    env = AirHockeyEnv(max_steps=max_steps)
    all_returns = []
    all_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        steps = 0

        while not (done or truncated):
            # Random action from the environment's action space
            action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            steps += 1

        all_returns.append(episode_return)
        all_lengths.append(steps)

        print(
            f"Episode {ep + 1}/{num_episodes} "
            f"- Return: {episode_return:.3f}, Steps: {steps}"
        )

    env.close()

    all_returns = np.array(all_returns, dtype=np.float32)
    all_lengths = np.array(all_lengths, dtype=np.int32)

    print("\n=== Random Policy Baseline Summary ===")
    print(f"Episodes:       {num_episodes}")
    print(f"Avg return:     {all_returns.mean():.4f}")
    print(f"Std return:     {all_returns.std():.4f}")
    print(f"Avg ep length:  {all_lengths.mean():.2f}")
    print(f"Min/Max length: {all_lengths.min()} / {all_lengths.max()}")

    return all_returns, all_lengths


if __name__ == "__main__":
    run_random_baseline(num_episodes=50, max_steps=500, seed=0)
