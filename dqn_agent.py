import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, obs_dim, capacity=100_000):
        self.capacity = capacity
        self.obs_dim = obs_dim

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        idx = self.ptr
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s2
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            states=self.states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_states=self.next_states[idxs],
            dones=self.dones[idxs],
        )
        return batch


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        min_buffer: int = 5_000,
        target_update_freq: int = 1_000,
        device: str | None = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.target_update_freq = target_update_freq

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Networks
        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(obs_dim, capacity=buffer_capacity)

        self.train_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        ε-greedy over Q(s,a).
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(
            0
        )  # (1, obs_dim)
        with torch.no_grad():
            q_vals = self.q_net(state_t)  # (1, n_actions)
        action = int(torch.argmax(q_vals, dim=1).item())
        return action

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.add(state, action, reward, next_state, done)

    def train_step(self):
        """
        One gradient update from a minibatch.
        """
        if self.buffer.size < self.min_buffer:
            return None

        batch = self.buffer.sample(self.batch_size)

        states = torch.tensor(batch["states"], device=self.device)
        actions = torch.tensor(batch["actions"], device=self.device).long()
        rewards = torch.tensor(batch["rewards"], device=self.device)
        next_states = torch.tensor(batch["next_states"], device=self.device)
        dones = torch.tensor(batch["dones"], device=self.device)

        # Q(s,a)
        q_vals = self.q_net(states)  # (B, n_actions)
        q_sa = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_net(next_states)  # (B, n_actions)
            max_q_next, _ = torch.max(q_next, dim=1)
            target = rewards + self.gamma * (1.0 - dones) * max_q_next

        loss = nn.MSELoss()(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1

        # Periodically copy weights to target
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
