from actor import Actor
from critic import Critic
from tasks.task import Task
from replay_buffer import ReplayBuffer
from ounoise import OUNoise

class DDPG_Agent():
    def __init__(self, task)
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        self.actor_learninig_rate = 0.01
        self.crotoc_learning_rate = 0.01
        
        # Initialize actors and critics
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        # Replay buffer
        self.buffer_size = 100000
        self.batch_size = 64
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # for soft update of target parameters
        
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state
    
    def act(self, state):
        feed = {self.actor_local.inputs_: np.reshape(state, [-1, self.state_size])}
        action = sess.run(self.actor_local.actions, feed_dict=feed)
        return list(action + self.noise.sample())
    
    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.buffer.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in buffer
        if len(self.buffer) > self.batch_size:
            experiences = self.buffer.sample()
            self.learn()

        # Roll over last state and action
        self.last_state = next_state
        
    def lear(self):
        experiences = self.buffer.sample()
        
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        
        # Get predicted next-state actions and Q values from target models
        actions_next = sess.run(self.actor_target.actions, feed_dict ={
            self.actor_target.inputs_: next_states})
        Q_targets_next = sess.run(self.critic_target.Q_value, feed_dict ={
            self.critic_target.state_: next_states,
            self.critic_target.action_: actions_next})
        

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        # train critic
        _, action_value_gradients = sess.run([self.critic_local.opt, self.critic_local.action_value_gradients], feed_dict ={
            self.critic_local.state_: states,
            self.critic_local.action_: actions,
            self.critic_local.label_: Q_targets})

        # Train actor model (local)
        action_value_gradients = np.reshape(action_value_gradients, (-1, self.action_size))
        sess.run(self.actor_local.opt, feed_dict={
            self.actor_local.inputs_: states,
            self.actor_local.action_values_gradient_: action_value_gradients
        })
        
#          # Soft-update target models
#         self.soft_update(self.critic_local.model, self.critic_target.model)
#         self.soft_update(self.actor_local.model, self.actor_target.model)   

#     def soft_update(self, local_model, target_model):
#         """Soft update model parameters."""
#         local_weights = np.array(local_model.get_weights())
#         target_weights = np.array(target_model.get_weights())

#         assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

#         new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
#         target_model.set_weights(new_weights)
        
        