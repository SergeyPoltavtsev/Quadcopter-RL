import numpy as np
import tensorflow as tf

class Critic():
    def __init__(self, state_size, action_size, learning_rate = 0.01):
        self.state_size = state_size
        self.action_size = acrion_size
        self.learning_rate = learning_rate
        self.build_model()
    
    def build_model(self):
        with tf.variable_scope("Critic")
            self.state_ = tf.placeholder(tf.float32, shape=[None, self.state_size], name="state")
            self.action_ = tf.placeholder(tf.float32, shape=[None, self.action_size], name="action")
            self.label_ = tf.placeholder(tf.float32, shape=[None], name="action_value_label")
            
            # Define network
            self.fc1_states = tf.contrib.layers.fully_connected(self.state_, 32)
            self.fc2_states = tf.contrib.layers.fully_connected(self.fc1_states, 64)
            
            self.fc1_actions = tf.contrib.layers.fully_connected(self.action_, 32)
            self.fc2_actions = tf.contrib.layers.fully_connected(self.fc1_actions, 64)
            
            # Combine two pathways for the input state and action
            self.fc3 = tf.add(self.fc2_states, self.fc2_actions)
            self.fc3 = tf.nn.relu(self.fc3)
            
            # Estimate action value
            self.Q_value = tf.contrib.layers.fully_connected(self.fc3, 1, activation_fn=None)
            
            # Define action value gradients which will be passed to the actor loss function.
            self.action_value_gradients = tf.gradients(self.Q_value, self.action_)
            
            # Define loss
            mse_loss = tf.losses.mean_squared_error(self.label_, self.Q_value)
            self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(mse_loss)
            
        