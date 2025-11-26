import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from air_hockey_env import AirHockeyEnv


# Discrete action set (reuse from Week 3 style)
def create_discrete_actions():
    """
    Map a small discrete action set to continuous accelerations
    in env action space ([-1, 1] in each dimension).

    Actions:
        0: stay
        1: left
        2: right
        3: up
        4: down
    """
    ACTION_MAG = 1.0

    actions = np.array(
        [
            [0.0, 0.0],  # stay
            [-ACTION_MAG, 0.0],  # left
            [ACTION_MAG, 0.0],  # right
            [0.0, ACTION_MAG],  # up
            [0.0, -ACTION_MAG],  # down
        ],
        dtype=np.float32,
    )
    return actions


# Dynamics model: NN predicting Δstate
class DynamicsModel(nn.Module):
    """
    Simple feedforward neural network dynamics model.

    Input:  state (8D) + action (2D) -> 10D
    Output: delta_state (8D)
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        """
        state: (B, state_dim)
        action: (B, action_dim)
        returns delta_state: (B, state_dim)
        """
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return delta


# Data collection from real env
def collect_data(
    env: AirHockeyEnv,
    num_episodes: int = 100,
    max_steps: int = 200,
    policy: str = "random",
    seed: int | None = 0,
):
    """
    Collect (s, a, s_next) tuples from the real environment.

    For Week 4, we can use a random policy as the behavior policy.
    """

    if seed is not None:
        np.random.seed(seed)

    discrete_actions = create_discrete_actions()
    state_list = []
    action_list = []
    next_state_list = []

    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        steps = 0

        while not (done or truncated) and steps < max_steps:
            if policy == "random":
                action_cont = env.action_space.sample()
            elif policy == "random_discrete":
                idx = np.random.randint(discrete_actions.shape[0])
                action_cont = discrete_actions[idx]
            else:
                # default to random continuous
                action_cont = env.action_space.sample()

            next_state, reward, done, truncated, info = env.step(action_cont)

            state_list.append(state.astype(np.float32))
            action_list.append(action_cont.astype(np.float32))
            next_state_list.append(next_state.astype(np.float32))

            state = next_state
            steps += 1

    states = np.stack(state_list, axis=0)
    actions = np.stack(action_list, axis=0)
    next_states = np.stack(next_state_list, axis=0)

    return states, actions, next_states


# Train the dynamics model
def train_dynamics_model(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    num_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Train DynamicsModel to predict next_state - state.
    """

    device = torch.device(device)
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    model = DynamicsModel(state_dim=state_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Prepare tensors
    states_t = torch.from_numpy(states).float().to(device)
    actions_t = torch.from_numpy(actions).float().to(device)
    next_states_t = torch.from_numpy(next_states).float().to(device)

    deltas_t = next_states_t - states_t  # Δs target

    num_samples = states.shape[0]
    num_batches = max(1, num_samples // batch_size)

    for epoch in range(num_epochs):
        perm = torch.randperm(num_samples, device=device)
        states_shuffled = states_t[perm]
        actions_shuffled = actions_t[perm]
        deltas_shuffled = deltas_t[perm]

        epoch_loss = 0.0

        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, num_samples)

            s_batch = states_shuffled[start:end]
            a_batch = actions_shuffled[start:end]
            d_batch = deltas_shuffled[start:end]

            optimizer.zero_grad()
            pred_delta = model(s_batch, a_batch)
            loss = loss_fn(pred_delta, d_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= num_batches
        if (epoch + 1) % 10 == 0:
            print(f"[Dynamics] Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    return model


# MPC via random shooting with learned dynamics
def mpc_action_random_shooting(
    model: DynamicsModel,
    current_state: np.ndarray,
    horizon: int,
    num_samples: int,
    discrete_actions: np.ndarray,
    device: str = "cpu",
):
    """
    Use random shooting MPC with the learned model to choose an action.

    current_state: (state_dim,)
    Returns: first action (2D continuous) of the best action sequence.
    """
    device = torch.device(device)
    model.eval()

    state_dim = current_state.shape[0]
    n_actions = discrete_actions.shape[0]

    # Tile the current state for all sampled trajectories
    s0 = torch.from_numpy(current_state).float().to(device)
    s0_batch = s0.unsqueeze(0).repeat(num_samples, 1)  # (num_samples, state_dim)

    # Sample sequences of discrete action indices
    # shape: (num_samples, horizon)
    action_indices = np.random.randint(
        low=0, high=n_actions, size=(num_samples, horizon)
    )

    # Convert to torch actions
    actions_seq = torch.from_numpy(discrete_actions[action_indices]).float().to(device)
    # actions_seq shape: (num_samples, horizon, action_dim)

    # Rollout using the learned dynamics model
    states = s0_batch
    total_rewards = torch.zeros(num_samples, device=device)

    with torch.no_grad():
        for t in range(horizon):
            a_t = actions_seq[:, t, :]  # (num_samples, action_dim)
            delta = model(states, a_t)
            states = states + delta  # predicted next state

            # Reward function for planning (separate from env reward):
            # Encourage paddle to be close to puck and puck y near "top" goal.
            x_p, y_p, vx_p, vy_p, x_pad, y_pad, vx_pad, vy_pad = (
                states[:, 0],
                states[:, 1],
                states[:, 2],
                states[:, 3],
                states[:, 4],
                states[:, 5],
                states[:, 6],
                states[:, 7],
            )

            # Distance between paddle and puck
            dist = torch.sqrt((x_p - x_pad) ** 2 + (y_p - y_pad) ** 2)

            # Reward components
            #  - get closer to the puck (negative distance)
            #  - encourage puck to move toward positive y (e.g., opponent goal at top)
            reward_proximity = -dist
            reward_puck_y = 0.1 * y_p  # higher y is better
            reward_speed_upwards = 0.1 * vy_p

            r_t = reward_proximity + reward_puck_y + reward_speed_upwards

            total_rewards += r_t

    # Choose the best sequence
    best_idx = torch.argmax(total_rewards).item()
    best_first_action_idx = action_indices[best_idx, 0]
    best_first_action = discrete_actions[best_first_action_idx]

    return best_first_action


def run_mbrl_mpc_demo(
    data_episodes: int = 100,
    train_epochs: int = 50,
    mpc_episodes: int = 10,
    horizon: int = 15,
    num_samples: int = 256,
    max_steps: int = 200,
    device: str = "cpu",
):
    # 1. Collect data
    print("Collecting data from real environment...")
    env = AirHockeyEnv(max_steps=max_steps)
    states, actions, next_states = collect_data(
        env,
        num_episodes=data_episodes,
        max_steps=max_steps,
        policy="random_discrete",
        seed=0,
    )
    print(f"Collected {states.shape[0]} transitions.")

    # 2. Train dynamics model
    print("Training dynamics model...")
    model = train_dynamics_model(
        states=states,
        actions=actions,
        next_states=next_states,
        num_epochs=train_epochs,
        batch_size=256,
        lr=1e-3,
        device=device,
    )

    # 3. Use MPC with learned model to control real env
    print("Running MPC-controlled episodes with learned dynamics...")
    discrete_actions = create_discrete_actions()
    returns = []

    for ep in range(mpc_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            action_cont = mpc_action_random_shooting(
                model=model,
                current_state=state,
                horizon=horizon,
                num_samples=num_samples,
                discrete_actions=discrete_actions,
                device=device,
            )

            next_state, reward, done, truncated, info = env.step(action_cont)
            total_reward += reward
            state = next_state
            steps += 1

            if steps >= max_steps:
                truncated = True

        returns.append(total_reward)
        print(
            f"[MPC] Episode {ep+1}/{mpc_episodes} - Return: {total_reward:.4f}, Steps: {steps}"
        )

    env.close()
    returns = np.array(returns, dtype=np.float32)
    print("\n=== MPC with Learned Dynamics Finished ===")
    print(f"Mean return over {mpc_episodes} episodes: {returns.mean():.4f}")


if __name__ == "__main__":
    run_mbrl_mpc_demo(
        data_episodes=100,
        train_epochs=50,
        mpc_episodes=5,
        horizon=15,
        num_samples=256,
        max_steps=200,
        device="cpu",
    )
