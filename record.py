import numpy as np
import imageio.v2 as imageio
from air_hockey_mjenv import AirHockeyMujocoEnv
from QLearning import LinearQAgent, create_discrete_actions


# Load pre-trained agents
def load_final_agent():
    W = np.load("trained_weights.npy")
    agent = LinearQAgent(n_features=15, n_actions=5)
    agent.W = W
    return agent


def load_best_agent():
    W = np.load("best_trained_weights.npy")
    agent = LinearQAgent(n_features=15, n_actions=5)
    agent.W = W
    return agent


# Recording helpers
def record_policy(env, agent, discrete_actions, episodes, filename, epsilon_eval=0.0):
    frames = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action_idx = agent.select_action(obs, epsilon_eval)
            action = discrete_actions[action_idx]

            obs, reward, done, truncated, info = env.step(action)
            frame = env.render()
            frames.append(frame)

    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved {filename}")


def record_random_discrete(env, discrete_actions, episodes, filename):
    frames = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = truncated = False

        while not (done or truncated):
            idx = np.random.randint(len(discrete_actions))
            action = discrete_actions[idx]
            obs, reward, done, truncated, info = env.step(action)
            frames.append(env.render())

    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved {filename}")


def main():
    discrete_actions = create_discrete_actions()

    # Create env for recording
    env = AirHockeyMujocoEnv(max_steps=500, render_mode="rgb_array")

    # Random baseline
    record_random_discrete(env, discrete_actions, episodes=3, filename="random.gif")

    # Load pre-trained agents
    final_agent = load_final_agent()
    best_agent = load_best_agent()

    # Record final greedy policy
    record_policy(
        env,
        final_agent,
        discrete_actions,
        episodes=5,
        filename="final_greedy.gif",
        epsilon_eval=0.0,
    )

    # Record best scoring agent
    record_policy(
        env,
        best_agent,
        discrete_actions,
        episodes=5,
        filename="best_policy.gif",
        epsilon_eval=0.0,
    )

    env.close()


if __name__ == "__main__":
    main()
