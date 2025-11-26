import numpy as np
from air_hockey_env import AirHockeyEnv

env = AirHockeyEnv()
obs, info = env.reset()

print("Initial obs:", obs)

for t in range(10):
    action = env.action_space.sample()  # random paddle acceleration
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t}: reward={reward:.4f}, obs={obs}")
    if terminated or truncated:
        print("Episode ended")
        break

env.close()
