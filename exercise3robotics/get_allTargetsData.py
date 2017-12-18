import numpy as np; np.random.seed(0)
from random import randrange
# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from createMaxGoalTrainingSet import hp_sets
# Initialization
opt = Options()

# different hyper param settings
hp_sets = hp_sets
#hp_sets.append({'change_tgt': True, 'tgt_y':2,'tgt_x':2})
#hp_sets.append({'change_tgt': True, 'tgt_y':25,'tgt_x':11})
#hp_sets.append({'change_tgt': True, 'tgt_y':25,'tgt_x':25})
#hp_sets.append({'change_tgt': True, 'tgt_y':2,'tgt_x':25})
#hp_sets.append({'change_tgt': True, 'tgt_y':20,'tgt_x':20})
#hp_sets.append({'change_tgt': True, 'tgt_y':6,'tgt_x':23})
#hp_sets.append({'change_tgt': True, 'tgt_y':16,'tgt_x':10})
#hp_sets.append({'change_tgt': True, 'tgt_y':12,'tgt_x':5})
#hp_sets.append({'change_tgt': True, 'tgt_y':9,'tgt_x':22})
#hp_sets.append({'change_tgt': True, 'tgt_y':17,'tgt_x':5})


states = np.zeros([opt.data_steps, opt.state_siz], float)
labels = np.zeros([opt.data_steps], int)
for i in hp_sets:
	print("i ",i)
	opt.change_tgt = i['change_tgt']
	opt.tgt_y = i['tgt_y']
	opt.tgt_x = i['tgt_x']
	
	sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
	

	# Note I am forcing the display to be off here to make data collection fast
	# you can turn it on again for debugging purposes
	opt.disp_on = False

	# 1. control loop
	if opt.disp_on:
	    win_all = None
	    win_pob = None
	epi_step = 0    # #steps in current episode
	nepisodes = 1   # total #episodes executed

	state = sim.newGame(opt.tgt_y, opt.tgt_x)
	for step in range(opt.data_steps):
	    if state.terminal or epi_step >= opt.early_stop:
	        epi_step = 0
	        nepisodes += 1
	        state = sim.newGame(opt.tgt_y, opt.tgt_x)
	    else:
	        state = sim.step() # will perform A* actions

	    # save data & label
	    print("Shapes: ",rgb2gray(state.pob).shape)
	    if(state.pob.shape[0] != 25):
	    	while(rgb2gray(state.pob).shape[0] != 25):
	    		print("fill 0",rgb2gray(state.pob).shape)
		    	#state.pob = np.expand_dims(state.pob, axis=0)
		    	row = np.zeros((1,25))
		    	state.pob = np.append(rgb2gray(state.pob),row, axis=0)
		if(state.pob.shape[1] != 25):
			while(rgb2gray(state.pob).shape[1] != 25):
				print("fill 1",rgb2gray(state.pob).shape)
				#state.pob = np.expand_dims(state.pob, axis=1)
		    	col = np.zeros((25,1))
		    	state.pob = np.append(rgb2gray(state.pob),col, axis=1)
	    states[step, :] = rgb2gray(state.pob).reshape(opt.state_siz)
	    labels[step]    = state.action

	    epi_step += 1

	    #if step % opt.prog_freq == 0:
	        #print(step)

	    if opt.disp_on:
	        if win_all is None:
	            import pylab as pl
	            pl.figure()
	            win_all = pl.imshow(state.screen)
	            pl.figure()
	            win_pob = pl.imshow(state.pob)
	        else:
	            win_all.set_data(state.screen)
	            win_pob.set_data(state.pob)
	        pl.pause(opt.disp_interval)
	        pl.draw()

# 2. save to disk
print('saving data ...')
np.savetxt(opt.states_fil, states, delimiter=',')
np.savetxt(opt.labels_fil, labels, delimiter=',')
print("states saved to " + opt.states_fil)
print("labels saved to " + opt.labels_fil)
