from policies.policy import Policy, Network
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax


@jax.jit(static_argnames=['apply_fn'])
def _sample_action(params, apply_fn, key, obs, std=0.1):
    mean = jnp.tanh(apply_fn(params, obs))
    action=jnp.clip(mean+std*jax.random.normal(key, mean.shape), -1, 1)
    return action

@jax.jit(static_argnames=['apply_fn'])
def _get_mean(params, apply_fn, obs):
    mean = jnp.tanh(apply_fn(params, obs))
    return mean


@jax.jit(static_argnames=['apply_fn'])
def _calc_pi_loss(apply_fn, params, obs, actions, returns):
    loss=-jnp.mean(returns*mean)
    return loss


@jax.jit(static_argnames=['apply_fn'])
def _calc_v_loss(apply_fn, params, obs, returns, actions, new_obs, dones, gamma):
    value = apply_fn(params, obs)
    loss=-jnp.mean((value - returns)**2)
    return loss


class A2C(Policy):
    def __init__(self, env, gamma=0.99, lr=0.001, dims=(64,64,1)):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.pi_net = Network(dims=dims)
        self.key = jax.random.PRNGKey(0)
        self.key, subkey = jax.random.split(self.key)
        params = self.pi_net.init(subkey, jnp.zeros((1, env.observation_space.shape[0])))
        self.pi_state = train_state.TrainState.create(
            apply_fn=self.pi_net.apply,
            params=params,
            tx=optax.adam(lr))
        self.v_net = Network(dims=dims)
        self.key, subkey = jax.random.split(self.key)
        params = self.v_net.init(subkey, jnp.zeros((1, env.observation_space.shape[0])))
        self.v_state = train_state.TrainState.create(
            apply_fn=self.v_net.apply,
            params=params,
            tx=optax.adam(lr))
    

    def act(self, obs, train=False):
        if train:
            self.key, subkey = jax.random.split(self.key)
            action = _sample_action(self.pi_state.params, self.pi_state.apply_fn, subkey, obs)
            return action
        else:
            action = _get_mean(self.pi_state.params, self.pi_state.apply_fn, obs)
            return action

    def train(self):
        obs, actions, returns, new_obs, dones = self.buffer.sample(batch_size)
        pi_loss = _calc_pi_loss(self.pi_state.apply_fn, self.pi_state.params, obs, actions, returns)
        v_loss = _calc_v_loss(self.v_state.apply_fn, self.v_state.params, obs, returns)
        self.pi_state, self.v_state = _update_step(self.pi_state, self.v_state, obs, actions, returns)
        return float(pi_loss), float(v_loss)

    def store(self, obs, action, reward, new_obs, done):
        self.buffer.add(obs, action, reward, new_obs, done)

    def store_episode(self):
        obs, _ = self.env.reset()
        terminated = False
        truncated = False