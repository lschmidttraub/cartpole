from policies.policy import Policy, Network, ReplayBuffer
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax


@jax.jit(static_argnames=['apply_fn'])
def _sample_action(params, apply_fn, key, obs, std=0.1):
    mean = apply_fn(params, obs)
    mean = jnp.tanh(mean)
    noise = std * jax.random.normal(key, mean.shape)
    action = jnp.clip(mean + noise, -1, 1)
    return action

@jax.jit(static_argnames=['apply_fn'])
def _get_mean(params, apply_fn, obs):
    mean = jnp.tanh(apply_fn(params, obs))
    return mean

def _calc_pi_loss(apply_pi_fn, pi_params, apply_q_fn, q_params, obs):
    actions = _get_mean(pi_params, apply_pi_fn, obs)
    q_values = apply_q_fn(q_params, jnp.concatenate([obs, actions], axis=-1))
    loss=-jnp.mean(q_values)
    return loss

def calc_q_loss(apply_q_fn, q_params, apply_q_target_fn, q_target_params, apply_pi_target_fn, pi_target_params, obs, actions, returns, new_obs, dones, gamma):
    q_values = apply_q_fn(q_params, jnp.concatenate([obs, actions], axis=-1))
    old_actions = _get_mean(pi_target_params, apply_pi_target_fn, new_obs)
    next_q_values = apply_q_target_fn(q_target_params, jnp.concatenate([new_obs, old_actions], axis=-1))
    y = returns + gamma * (1 - dones) * next_q_values
    loss = jnp.mean((q_values - y)**2)
    return loss

@jax.jit
def _update_pi_step(pi_state, q_state, obs):
    grad_fn = jax.value_and_grad(_calc_pi_loss, argnums=1, has_aux=False)
    loss, grads = grad_fn(pi_state.apply_fn, pi_state.params, q_state.apply_fn, q_state.params, obs)
    return pi_state.apply_gradients(grads=grads), loss

@jax.jit(static_argnames=['apply_q_target_fn', 'apply_pi_target_fn'])
def _update_q_step(q_state, q_target_params, apply_q_target_fn, pi_target_params, apply_pi_target_fn, obs, actions, returns, new_obs, dones, gamma):
    grad_fn = jax.value_and_grad(calc_q_loss, argnums=1, has_aux=False)
    loss, grads = grad_fn(q_state.apply_fn, q_state.params, apply_q_target_fn, q_target_params, apply_pi_target_fn, pi_target_params, obs, actions, returns, new_obs, dones, gamma)
    return q_state.apply_gradients(grads=grads), loss


@jax.jit
def polyak_update(target_params, source_params, tau):
    return jax.tree_util.tree_map(lambda x, y: tau * x + (1 - tau) * y, source_params, target_params)




class DDPG(Policy):
    def __init__(self, env, gamma=0.99, actor_lr=1e-3, critic_lr=3e-4, pi_dims=(64,64,1), q_dims=(64,64,1), tau=0.005, buffer_size=100000):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(buffer_size, env.observation_space.shape, env.action_space.shape)
        
        self.key = jax.random.PRNGKey(0)
        self.key, subkey = jax.random.split(self.key)
        self.pi_net = Network(dims=pi_dims)
        pi_params = self.pi_net.init(subkey, jnp.zeros((1, env.observation_space.shape[0])))
        self.pi_state = train_state.TrainState.create(
            apply_fn=self.pi_net.apply,
            params=pi_params,
            tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(actor_lr))
        )
        self.pi_target_params = pi_params
        self.q_net = Network(dims=q_dims)
        q_params = self.q_net.init(subkey, jnp.zeros((1, env.observation_space.shape[0] + env.action_space.shape[0])))
        self.q_state = train_state.TrainState.create(
            apply_fn=self.q_net.apply,
            params=q_params,
            tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(critic_lr))
        )
        self.q_target_params = q_params
        self.step=0


    def act(self, obs, train=False, std=0.1):
        if train:
            self.key, subkey = jax.random.split(self.key)
            action = _sample_action(self.pi_state.params, self.pi_state.apply_fn, subkey, obs, std=std)
            return action
        else:
            action = _get_mean(self.pi_state.params, self.pi_state.apply_fn, obs)
            return action

    def train(self, batch_size=64):
        if self.buffer.size < batch_size:
            return 0.0, 0.0
            
        self.step += 1
        obs, actions, returns, new_obs, dones = self.buffer.sample(batch_size)
        
        # We can reuse the apply functions from the main networks since the architecture is identical
        q_loss = -1.0
        if self.step % 2 == 0:
            self.q_state, q_loss = _update_q_step(
                self.q_state, 
                self.q_target_params, 
                self.q_state.apply_fn, 
                self.pi_target_params, 
                self.pi_state.apply_fn, 
                obs, actions, returns, new_obs, dones, self.gamma
            )
        self.pi_state, pi_loss = _update_pi_step(self.pi_state, self.q_state, obs)

        self.q_target_params = polyak_update(self.q_target_params, self.q_state.params, self.tau)
        self.pi_target_params = polyak_update(self.pi_target_params, self.pi_state.params, self.tau)

        return float(pi_loss), float(q_loss)

    def store(self, obs, action, reward, new_obs, done):
        self.buffer.add(obs, action, reward, new_obs, done)

    def store_episode(self, random=False):
        obs, _ = self.env.reset()
        terminated = False
        truncated = False
        episode_return = 0
        while not (terminated or truncated):
            if random:
                action = self.env.action_space.sample()
            else:
                action = self.act(obs, train=True)
                action = np.array(action) # Convert JAX array to numpy for env
                
            new_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.store(obs, action, reward, new_obs, terminated)
            obs = new_obs
            episode_return += reward
        return episode_return
