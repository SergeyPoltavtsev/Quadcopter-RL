import numpy as np
import tensorflow as tf

class Actor():
    def __init__(self, state_size, action_size, action_low, action_high, name="Actor", learning_rate=0.01)
        """
        Creates an instance of the actor.

        Args:
            state_size (int): Dimension of the state.
            action_size (int): Dimention of the action.
            action_low (int): Min value of each action dimension
            action_high (int): Max value of each action dimension
            name (string): variable scope name.
            learning_rate (float): learning rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = action_high - action_low
        self.name = name
        self.learning_rate = learning_rate
        
        self.build_model()
    
    def build_model(self):
        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, shape=[None, self.state_size], "inputs")
            
            # hidden layers, relu is default activation
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, 32)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 64)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 32)
            
            # output
            self.sigmoid_output = tf.contrib.layers.fully_connected(self.fc3, self.action_size, activation_fn=tf.nn.sigmoid, "sigmoid_output")
            
            # scale output
            self.actions = self.sigmoid_output * self.action_range + self.action_low
            
            # The policy gradient will be optimized with respect to NN weights 
            # therefore we have to pass the action value gradients an input parameter 
            # and not just action values.
            self.action_values_gradient_ = tf.input(tf.float32, shape=[None, self.state_size], "action_values_gradient")
            self.loss = tf.reduce_mean(-self.actions*self.action_values_gradient_)
            
            self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)