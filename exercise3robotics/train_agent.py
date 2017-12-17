import numpy as np
import tensorflow as tf

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable
# my agent class
from agent import Agent

# Initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

# Train agent and save parameters
train_data = trans.get_train()
valid_data = trans.get_valid()

agent = Agent()
accuracy = agent.train(train_data, valid_data)
print("Achieved validation accuracy of %f" % accuracy)
