import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf
import numpy as np


# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
from agent import Agent

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000 #DEBUG make smaller?!
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None


agent = Agent()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
#saver.restore(sess, "./data/policies.ckpt")

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 1 * 10**6 + 1
epi_step = 0
nepisodes = 0
solved_episodes = 0
loss = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
for step in range(steps):
    if step % 1000 == 0:
        save_path = saver.save(sess, "./data/policies.ckpt")
        print("Model saved in file: %s" % save_path)

    if state.terminal or epi_step >= opt.early_stop:
        print("\rEpisode {} ({}, {}), solved {}".format(nepisodes, epi_step, loss, solved_episodes))
        #agent.debug(sess)
        if epi_step < opt.early_stop:
            solved_episodes+=1

        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
        agent.clear_hist()

    # simulation step
    epi_step+=1
    action = agent.make_epsilon_greedy_action(sess, rgb2gray(state.pob).reshape(opt.state_siz))#state_with_history.reshape(-1))
    next_state = sim.step(action)

    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    action_onehot = trans.one_hot_action(action)
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state
    
    # train
    state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
    loss = agent.train(sess, state_batch, action_batch, next_state_batch, reward_batch, terminal_batch)
    if loss > 100:
        print("Dangerous loss")
        quit()
    #plot
    if opt.disp_on:
        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()
        plt.savefig('loss.png')