# watch_mujoco.py
import time
import numpy as np
from air_hockey_mjenv import AirHockeyMujocoEnv


def watch_random(num_episodes: int = 1, max_steps: int = 200):
    """
    Simple viewer using the env's matplotlib-based render().
    Good enough to visually check behavior or a trained agent.
    """
    env = AirHockeyMujocoEnv(render_mode="human", max_steps=max_steps)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0

        print(f"\n=== Episode {ep + 1}/{num_episodes} ===")

        while not (done or truncated) and steps < max_steps:
            # TODO: swap this out for your trained agent's action later
            action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)

            # Render every 2nd or 3rd step to keep it responsive
            if steps % 5 == 0:
                env.render()

            steps += 1

        print(f"Episode {ep + 1} finished after {steps} steps, info={info}")

    env.close()


if __name__ == "__main__":
    watch_random(num_episodes=1, max_steps=200)
