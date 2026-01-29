from abc import ABC, abstractmethod
import flax.linen as nn
from typing import Sequence 
import jax.numpy as jnp
import numpy as np

class Policy(ABC):
    def __init__(self, env, gamma=0.99, lr=0.001):
        self.env = env
        self.gamma=gamma
        self.lr = lr

    @abstractmethod
    def act(self, obs, train=False):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def store(self, transition):
        pass

    @abstractmethod
    def store_episode(self):
        pass

class GaussianNetwork(nn.Module):
    dims: Sequence[int]

    @nn.compact
    def __call__(self, obs):
        x = obs
        for dim in self.dims[:-1]:
            x = nn.relu(nn.Dense(dim)(x))
        x = nn.Dense(self.dims[-1])(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.dims[-1],))
        return x, log_std

class Network(nn.Module):
    dims: Sequence[int]

    @nn.compact
    def __call__(self, obs):
        x = obs
        for dim in self.dims[:-1]:
            x = nn.relu(nn.Dense(dim)(x))
        x = nn.Dense(self.dims[-1], kernel_init=nn.initializers.orthogonal(0.01))(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs]
        )

class RolloutBuffer:
    def __init__(self, obs_shape, action_shape):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_obs.append(next_obs)
        self.dones.append(done)

    def _empty(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []


    def sample(self):
        if self.obs is None:
            raise Exception("Replay buffer is empty")
        res = (
            jnp.array(self.obs),
            jnp.array(self.actions),
            jnp.array(self.rewards),
            jnp.array(self.next_obs),
            jnp.array(self.dones),
        )
        self._empty()
        return res
