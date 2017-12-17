import numpy as np
import tensorflow as tf

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable
# my agent class
from agent import Agent
from time import time

# Initialization
opt = Options()

# different hyper param settings
hp_sets = []
hp_sets.append({'change_tgt': True, 'tgt_y':2,'tgt_x':2})
hp_sets.append({'change_tgt': True, 'tgt_y':25,'tgt_x':11})
print("go:",hp_sets)
for i in hp_sets:
	print("i ",i)
	opt.change_tgt = i['change_tgt']
	opt.tgt_y = i['tgt_y']
	opt.tgt_x = i['tgt_x']
	print(opt)

	#continue
	sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
	trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
	                         opt.minibatch_size, opt.valid_size,
	                         opt.states_fil, opt.labels_fil)

	# Train agent and save parameters
	train_data = trans.get_train()
	valid_data = trans.get_valid()

	agent = Agent()
	accuracy = agent.train(train_data, valid_data)
	print("Achieved validation accuracy of %f for target at {},{}".format(accuracy,opt.tgt_x,opt.tgt_y))

	# 1. control loop
	if opt.disp_on:
		win_all = None
		win_pob = None
	epi_step = 0    # #steps in current episode
	nepisodes = 0   # total #episodes executed
	nepisodes_solved = 0

	# start a new game
	state = sim.newGame(opt.tgt_y, opt.tgt_x)
	for step in range(opt.eval_steps):
		print("\rStep {}/{}".format(step,opt.eval_steps ))
		# check if episode ended
		if state.terminal or epi_step >= opt.early_stop:
		    epi_step = 0
		    nepisodes += 1
		    if state.terminal:
		        nepisodes_solved += 1
		    # start a new game
		    state = sim.newGame(opt.tgt_y, opt.tgt_x)
		else:
		    action = agent.compute_next_action(state)
		    state = sim.step(action)

		    epi_step += 1

		if state.terminal or epi_step >= opt.early_stop:
		    epi_step = 0
		    nepisodes += 1
		    if state.terminal:
		        nepisodes_solved += 1
		    # start a new game
		    state = sim.newGame(opt.tgt_y, opt.tgt_x)

		if step % opt.prog_freq == 0:
		    print(step)

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

		# 2. calculate statistics
		print(float(nepisodes_solved) / float(nepisodes))
		# 3. TODO perhaps  do some additional analysis
		# TODO: would be interesting to compare steps taken by A* and steps taken by agent
