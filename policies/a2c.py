from policies.policy import Policy, Network, GaussianNetwork, RolloutBuffer
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

@jax.jit(static_argnames=['apply_fn'])
def _sample_action(params, apply_fn, key, obs):
    mean, log_std = apply_fn(params, obs)
    std = jnp.exp(log_std)
    action = mean + std*jax.random.normal(key, mean.shape)
    return action

@jax.jit(static_argnames=['apply_fn'])
def _get_mean(params, apply_fn, obs):
    mean, _ = apply_fn(params, obs)
    return mean

@jax.jit
def _compute_returns(reward_buffer, gamma):
    # Assumes exactly one episode has been simulated
    def scan_fn(G, r):
        # g_{t:T} = \gamma g_{t+1:T} + r_t
        G *= gamma
        G += r
        return G, G
    _, Gs = jax.lax.scan(scan_fn, 0, reward_buffer[::-1])
    return Gs[::-1]

@jax.jit
def _update_step(pi_state, v_state, obs, actions, returns, entropy_coeff=1e-3):
    # advantage calculated based on values
    values = v_state.apply_fn(v_state.params, obs).squeeze()
    advantages = returns - values
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    # Actor loss
    def pi_loss_fn(pi_params):
        mean, log_std = pi_state.apply_fn(pi_params, obs)
        log_std = jnp.clip(log_std, -20, 2)
        var = jnp.exp(2*log_std)
        
        logprobs = -0.5 * ((actions - mean) ** 2) / var - log_std - 0.5 * jnp.log(2 * jnp.pi)
        # Avoid dimension mismatch
        logprobs=logprobs.sum(-1)
        # Add entropy term (typical in practice)
        entropy=0.5*(1+jnp.log(2*jnp.pi)+2*log_std).sum(-1)
        # We don't scale by gamma**t because of vanishing gradients
        loss = -jnp.mean(logprobs * advantages + entropy*entropy_coeff)
        return loss

    pi_grad_fn = jax.value_and_grad(pi_loss_fn)
    pi_loss, pi_grads = pi_grad_fn(pi_state.params)
    new_pi_state = pi_state.apply_gradients(grads=pi_grads)

    # Critic loss
    def v_loss_fn(v_params):
        v_pred = v_state.apply_fn(v_params, obs).squeeze()
        loss = jnp.mean((v_pred - returns) ** 2)
        return loss

    v_grad_fn = jax.value_and_grad(v_loss_fn)
    v_loss, v_grads = v_grad_fn(v_state.params)
    new_v_state = v_state.apply_gradients(grads=v_grads)

    return new_pi_state, new_v_state, pi_loss, v_loss

class A2C(Policy):
    def __init__(self, env, gamma=0.99, lr=0.001, dims=(64,64,1)):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        
        self.buffer = RolloutBuffer(env.observation_space.shape, env.action_space.shape)
        
        # Initialize Actor (Policy) Network
        self.pi_net = GaussianNetwork(dims=dims)
        self.key = jax.random.PRNGKey(0)
        self.key, subkey = jax.random.split(self.key)
        
        obs_shape = env.observation_space.shape
        params = self.pi_net.init(subkey, jnp.zeros((1, obs_shape[0])))
        self.pi_state = train_state.TrainState.create(
            apply_fn=self.pi_net.apply,
            params=params,
            tx=optax.adam(lr)
        )
        
        # Initialize Critic (Value) Network
        # Critic output dim should be 1
        v_dims = list(dims)
        v_dims[-1] = 1
        self.v_net = Network(dims=v_dims)
        self.key, subkey = jax.random.split(self.key)
        params = self.v_net.init(subkey, jnp.zeros((1, obs_shape[0])))
        self.v_state = train_state.TrainState.create(
            apply_fn=self.v_net.apply,
            params=params,
            tx=optax.adam(lr)
        )

    def act(self, obs, train=False):
        if train:
            self.key, subkey = jax.random.split(self.key)
            action = _sample_action(self.pi_state.params, self.pi_state.apply_fn, subkey, obs)
            return action
        else:
            action = _get_mean(self.pi_state.params, self.pi_state.apply_fn, obs)
            return action

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def store_episode(self):
        obs, _ = self.env.reset()
        terminated = False
        truncated = False
        episode_return = 0
        while not (terminated or truncated):
            action = self.act(obs, train=True)
            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.store(obs, action, reward, new_obs, terminated)
            obs = new_obs
            episode_return += reward
        return episode_return

    def train(self):
        # A2C is on-policy, so we use the full collected batch usually.
        # We ignore batch_size argument if passed (compatibility with train.py)

        # Compute returns (Monte Carlo)
        obs, actions, rewards, next_obs, dones = self.buffer.sample()
        G = _compute_returns(jnp.array(rewards), self.gamma)
        
        self.pi_state, self.v_state, pi_loss, v_loss = _update_step(
            self.pi_state, self.v_state, obs, actions, G
        )
        
        # Clear buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        
        return float(pi_loss), float(v_loss)
