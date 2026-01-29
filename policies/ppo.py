from policies.policy import Policy, Network, GaussianNetwork, RolloutBuffer
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import jax.scipy.stats as jstats
import numpy as np


@jax.jit(static_argnames=['apply_fn'])
def _get_mean(params,apply_fn, obs):
    mean, _ = apply_fn(params, obs);
    return mean

@jax.jit(static_argnames=['apply_fn'])
def _sample_action(params, apply_fn, key, obs):
    mean, log_std = apply_fn(params, obs)
    std = jnp.exp(log_std)
    action = mean + std*jax.random.normal(key, mean.shape)
    return action
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

# def kl_divergence(mu_p, var_p, mu_q, var_q):
#     return 0.5*(jnp.log(var_q)-jnp.log(var_p)+var_p/var_q+(mu_p-mu_q)**2/var_q-1)


def get_update_fn(old_pi_state, old_v_state, obs, actions, returns, epsilon, entropy_coef=1e-3):
    # Precompute constants for this PPO batch (fixed across update epochs).
    old_mean, old_log_std = old_pi_state.apply_fn(old_pi_state.params, obs)
    old_logp = jstats.norm.logpdf(actions, old_mean, jnp.exp(old_log_std)).sum(-1)

    old_values = old_v_state.apply_fn(old_v_state.params, obs).squeeze()
    advantages = returns - old_values
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    advantages = jax.lax.stop_gradient(advantages)

    @jax.jit
    def update_fn(pi_state, v_state):
        # Actor update: gradient w.r.t. pi_state.params only.
        def pi_loss_fn(pi_params):
            mean, log_std = pi_state.apply_fn(pi_params, obs)
            logp = jstats.norm.logpdf(actions, mean, jnp.exp(log_std)).sum(-1)
            ratio = jnp.exp(logp - old_logp)

            unclipped = ratio * advantages
            clipped = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
            clip_obj = jnp.minimum(unclipped, clipped)
            policy_loss = -jnp.mean(clip_obj)

            entropy = 0.5 * (1 + jnp.log(2 * jnp.pi) + 2 * log_std).sum(-1)
            entropy_bonus = jnp.mean(entropy)
            return policy_loss - entropy_coef * entropy_bonus

        # Critic update: gradient w.r.t. v_state.params only.
        def v_loss_fn(v_params):
            values = v_state.apply_fn(v_params, obs).squeeze()
            return jnp.mean((returns - values) ** 2)

        pi_loss, pi_grads = jax.value_and_grad(pi_loss_fn)(pi_state.params)
        v_loss, v_grads = jax.value_and_grad(v_loss_fn)(v_state.params)

        new_pi_state = pi_state.apply_gradients(grads=pi_grads)
        new_v_state = v_state.apply_gradients(grads=v_grads)
        return new_pi_state, new_v_state, pi_loss, v_loss

    return update_fn


class PPO(Policy):
    def __init__(self, env, gamma=0.99, lr=1e-3, dims=(64,64,1)):
        super().__init__(env,gamma,lr)

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
        
    def train(self, iterations=10):
        obs, actions, rewards, _, _ = self.buffer.sample()
        returns = _compute_returns(jnp.array(rewards), self.gamma)
        update_fn = get_update_fn(self.pi_state, self.v_state, obs, actions, returns, epsilon=0.1)
        pi_losses = []
        v_losses = []
        for _ in range(iterations):
            new_pi_state, new_v_state, pi_loss, v_loss = update_fn(self.pi_state, self.v_state)
            pi_losses.append(pi_loss)
            v_losses.append(v_loss)
            self.pi_state = new_pi_state
            self.v_state = new_v_state
        return float(np.mean(pi_losses)), float(np.mean(v_losses))