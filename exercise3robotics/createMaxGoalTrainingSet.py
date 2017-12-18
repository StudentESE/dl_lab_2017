import numpy as np
from random import randrange
# custom modules
from utils import State
from maps import maps

# different hyper param settings
hp_sets = []
x = 0
y = 0
for row in maps[0]:
	x += 1
	y = 0
	for col in row:
		y += 1
		if col == 0:
			#print(x,y)
			hp_sets.append({'change_tgt': True, 'tgt_y':y,'tgt_x':x})

#print(hp_sets)