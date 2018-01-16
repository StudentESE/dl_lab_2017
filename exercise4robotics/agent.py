import tensorflow as tf
import numpy as np
from utils import Options

opt = Options()

class Agent():

    def __init__(self):
        # Store the history of the observed states
        self.history = np.zeros((opt.hist_len, opt.state_siz))

        # Build the tensorflow graph
        self.states = tf.placeholder(tf.float32, shape=(None, opt.hist_len*opt.state_siz))
        self.actions = tf.placeholder(tf.float32, shape=(None, opt.act_num))

        self.states_next = tf.placeholder(tf.float32, shape=(None, opt.hist_len*opt.state_siz))
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1))
        self.terminals = tf.placeholder(tf.float32, shape=(None, 1))

        self.Q = self.forward_pass(self.states)
        Q_next =  self.forward_pass(self.states_next)

        best_actions_next = tf.one_hot(tf.argmax(Q_next, axis=1), 5)

        # Calculate the loss
        self.loss = self.Q_loss(
            self.Q,
            self.actions,
            Q_next,
            best_actions_next,
            self.rewards,
            self.terminals)

        # Setup an optimizer in tensorflow to minimize the loss
        self.train_step = tf.train.AdamOptimizer(opt.learning_rate).minimize(self.loss)

    def forward_pass(self, states):
        """
        TODO, defines structure of network, returns Q vector
        """
        input_layer = tf.reshape(states, [-1, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len])
        out_conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=opt.num_filters,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), #tf.random_uniform_initializer(0, 0.01),
            activation=tf.nn.relu)
        #initiliaze 0.01 (default is 0.1), 5000 steps

        out_conv2 = tf.layers.conv2d(
            inputs=out_conv1,
            filters=opt.num_filters,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), #tf.random_uniform_initializer(0, 0.01),
            activation=tf.nn.relu)

        flat_shape = int(out_conv2.shape[1]*out_conv2.shape[2]*out_conv2.shape[3])
        out_conv2_flat = tf.reshape(out_conv2, [-1, flat_shape])

        out_fcon1 = tf.layers.dense(
            inputs=out_conv2_flat,
            units=opt.num_units_linear_layer,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), #tf.random_uniform_initializer(0, 0.01),
            activation=tf.nn.relu)

        out_fcon2 = tf.layers.dense(
            inputs=out_fcon1,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), #tf.random_uniform_initializer(0, 0.01),
            units=opt.act_num)

        predictions = out_fcon2
        return predictions


    def Q_loss(self, Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
        """
        All inputs should be tensorflow variables!
        We use the following notation:
           N : minibatch size
           A : number of actions
        Required inputs:
           Q_s: a NxA matrix containing the Q values for each action in the sampled states.
                This should be the output of your neural network.
                We assume that the network implments a function from the state and outputs the
                Q value for each action, each output thus is Q(s,a) for one action
                (this is easier to implement than adding the action as an additional input to your network)
           action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                          (e.g. each row contains only one 1)
           Q_s_next: a NxA matrix containing the Q values for the next states.
           best_action_next: a NxA matrix with the best current action for the next state
           reward: a Nx1 matrix containing the reward for the transition
           terminal: a Nx1 matrix indicating whether the next state was a terminal state
           discount: the discount factor
        """
        # calculate: reward + discount * Q(s', a*),
        # where a* = arg max_a Q(s', a) is the best action for s' (the next state)
        target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
        # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
        #       use it as the target for Q_s
        target_q = tf.stop_gradient(target_q)
        # calculate: Q(s, a) where a is simply the action taken to get from s to s'
        selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
        loss = tf.reduce_sum(tf.square(selected_q - target_q))
        return loss

    def clear_hist(self):
        """
        TODO, clear if new episode starts
        """
        self.history[:] = 0

    def compute_q(self, sess, obs):
        def append_to_hist(obs):
            """
            Add observation to the history.
            """
            for i in range(self.history.shape[0]-1):
                self.history[i, :] = self.history[i+1, :]
            self.history[-1, :] = obs

        append_to_hist(obs)
        network_input = self.history.reshape(-1)
        q_ = sess.run(self.Q, feed_dict={self.states: [network_input]})[0]
        return q_

    def make_greedy_action(self, sess, obs):
        q_ = self.compute_q(sess, obs)
        best_action = np.argmax(q_)
        return best_action

    def make_epsilon_greedy_action(self, sess, obs, epsilon=0.95):
        best_action = self.make_greedy_action(sess, obs)
        probability = np.ones(opt.act_num, dtype=float) * epsilon / opt.act_num
        probability[best_action] += (1.0 - epsilon)

        action = np.random.choice(np.arange(opt.act_num), p=probability)
        return action

    def make_super_greedy_action(self, sess, obs):
        """
        TODO, makes greedy action but never "null" step (i.e does not stay in place)
        """
        q_ = self.compute_q(sess, obs)
        best_action = np.argmax(q_[1:len(q_)]) + 1
        return best_action

    def train(self, sess, state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
        """
        TODO, updates weights of network
        """
        dict = {
            self.states : state_batch,
            self.states_next : next_state_batch,
            self.actions : action_batch,
            self.rewards : reward_batch,
            self.terminals : terminal_batch}
        sess.run(self.train_step, feed_dict = dict)
        #DEBUG
        return sess.run(self.loss, feed_dict = dict)

    def debug(self, sess):
        network_input = self.history.reshape(-1)
        q_ = sess.run(self.Q, feed_dict={self.states: [network_input]})
        print(q_)
