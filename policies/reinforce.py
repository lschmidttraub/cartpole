from policies.policy import Policy, Network, GaussianNetwork
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import functools

@jax.jit(static_argnames=['apply_fn'])
def _sample_action(params, apply_fn, key, obs):
    mean, log_std = apply_fn(params, obs)
    std = jnp.exp(log_std)
    action=mean+std*jax.random.normal(key, mean.shape)
    return action, mean


@jax.jit(static_argnames=['apply_fn'])
def _get_mean(params, apply_fn, obs):
    mean, _ = apply_fn(params, obs)
    return mean

def _calc_loss(apply_fn, params,obs, actions, returns):
    mean, log_std = apply_fn(params, obs)
    var = jnp.exp(2*log_std)
    logprobs = -(actions - mean)**2/(2*var)-log_std-0.5*jnp.log(2*jnp.pi)
    logprobs = logprobs.sum(-1)
    loss=-jnp.mean(returns*logprobs)
    return loss

@jax.jit
def _update_step(state,obs,actions,returns):
    grad_fn = jax.value_and_grad(_calc_loss, argnums=1, has_aux=False)
    loss, grads = grad_fn(state.apply_fn, state.params,obs, actions, returns)
    return state.apply_gradients(grads=grads), loss

@jax.jit
def _compute_returns(reward_buffer, gamma):
    def scan_fn(G, r):
        # g_{t:T} = \gamma g_{t+1:T} + r_t
        G*=gamma
        G+=r
        return G, G
    _, Gs = jax.lax.scan(scan_fn, 0, reward_buffer[::-1])
    return Gs[::-1]

class Reinforce(Policy):
    def __init__(self, env, gamma=0.99, lr=0.001, dims=(64,64,1)):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.reward_buffer = []
        self.action_buffer = []
        self.obs_buffer = []
        self.action_history = []
        self.pi_net = GaussianNetwork(dims=dims)
        self.key = jax.random.PRNGKey(0)
        self.key, subkey = jax.random.split(self.key)
        params = self.pi_net.init(subkey, jnp.zeros((1, env.observation_space.shape[0])))
        self.pi_state = train_state.TrainState.create(
            apply_fn=self.pi_net.apply,
            params=params,
            tx=optax.adam(lr)
        )


    def act(self, obs, train=False):
        if train:
            self.key, subkey=jax.random.split(self.key)
            action, _ = _sample_action(self.pi_state.params, self.pi_state.apply_fn, subkey, obs)
            return action
        else:
            action = _get_mean(self.pi_state.params, self.pi_state.apply_fn, obs)
            return action

    def train(self):
        G = _compute_returns(jnp.array(self.reward_buffer), self.gamma)
        G = (G - G.mean()) / (G.std() + 1e-8)
        actions = jnp.array(self.action_buffer)
        obs = jnp.array(self.obs_buffer)
        self.pi_state, loss = _update_step(self.pi_state,obs, actions, G)
        self.obs_buffer=[]
        self.action_buffer=[]
        self.reward_buffer=[]
        return float(loss)

    def store(self, obs, action, reward):
        self.obs_buffer.append(np.asarray(obs)) 
        self.action_buffer.append(np.asarray(action))
        self.reward_buffer.append(np.asarray(reward))

    def store_episode(self):
        obs, _ = self.env.reset()
        terminated = False
        truncated = False
        episode_return = 0
        while not (terminated or truncated):
            action = self.act(obs, train=True)
            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.store(obs, action, reward)
            obs=new_obs
            episode_return += reward
        return episode_return
