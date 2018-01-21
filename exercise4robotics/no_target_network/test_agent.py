import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from agent import Agent

# initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

disp_on = False
if disp_on:
    win_all = None
    win_pob = None

agent = Agent()
sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "./data/policies.ckpt")

num_episodes = 300
solved_episodes = 0

# evaluate episodes and check how many are solved
for _ in range(num_episodes):
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    agent.clear_hist()
    epi_step = 0
    solved = False
    while not solved and epi_step <= opt.early_stop:
        if state.terminal:
            solved_episodes += 1
            solved = True

        # simulation step
        epi_step += 1
        action = agent.make_super_greedy_action(sess, rgb2gray(state.pob).reshape(opt.state_siz))
        state = sim.step(action)

        if disp_on:
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
print("Solved episodes: %f"%(solved_episodes/num_episodes))
