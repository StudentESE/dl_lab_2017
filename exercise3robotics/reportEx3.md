# Assignment 3 - Visual 
(Daniel B., Max Schlichting)

## INTRODUCTION
In this third assignment our task was to implement a CNN to perform visual planning. We were provided with a simulation framework and an A∗ planner. This A∗ planner was used to generate training data. The agent gets to see a local view of the map (plus some history) and has to predict the optimal action

## IMPLEMENTATION

The code stub provided to files to be modified by us. First we had to train our agent in train_agent.py, second we had to test its performance using test_agent.py. The performance of the agent is evaluated in terms of the percentage of episodes the agent was able to solve, i.e. the number of episodes where the agent found the goal divided by the total number of episodes.
We implemented to classes: A class agent.py which stores the agents neural network as well as its seen history. Furthermore the class agent.py provides a method to train the neural agents network as well as a method to predict the next step, given the current state and the agents history. If the history seen so far is less than the parameter hist_len, then the agent performs a random action in order to explore the environment, otherwise the agents neural network is used to predict the step. The second class implemented by us is the neural network network.py. It mainly reuses the code from the previous exercise (classification of the MNSIT dataset). As the input image is similar (almost the same size, greyscale) we kept the network architecture and only modified the second linear layer and the output layer. Instead of ten classes (digits) we only have to predict five actions ("nothing", "left", "right", "up", "down"). Hence we only need five outputs. The action mainly depends on eight direct neighbor cells of the robot, which is why we reduced the units of the second layer to eight. We also tried other
  1
parameters but eight units in the second layer showed a good performance. In addition to this minor changes of the architecture of the network we added code write the learned network weights to a file.

## EVALUATION

When we run the original map with the default goal we got 93% accuracy and 100/100 Episodes found the goal.
Changing the goal only one digit also works but more causes bad results. Means we have string overfitting which has to be answered in some way.
The fist approach was to add 10 goals for A* and train on the sum of episodes. The tests showed something like this:

```
Solved 0.0 Episodes for Goal at 2.0,2.0 with validation accuracy of 93.1999981403%
Solved 0.0 Episodes for Goal at 11.0,25.0 with validation accuracy of 93.1999981403%
Solved 0.0 Episodes for Goal at 25.0,25.0 with validation accuracy of 93.1999981403%
Solved 0.0 Episodes for Goal at 25.0,2.0 with validation accuracy of 93.1999981403%
Solved 0.1 Episodes for Goal at 20.0,20.0 with validation accuracy of 93.1999981403%
Solved 0.0 Episodes for Goal at 23.0,6.0 with validation accuracy of 93.1999981403%
Solved 0.0 Episodes for Goal at 10.0,16.0 with validation accuracy of 93.1999981403%
Solved 0.1 Episodes for Goal at 5.0,12.0 with validation accuracy of 93.1999981403%
Solved 0.0 Episodes for Goal at 22.0,9.0 with validation accuracy of 93.1999981403%
Solved 0.882352941176 Episodes for Goal at 5.0,17.0 with validation accuracy of 93.1999981403%
```

So 2 of 10 goals are found in 10% of the episodes and on in 83%.
The 7 of 10 never found episodes are bad which let us search for new ideas.
The current idea is to generate training data for all possible targets in the map and train the agent on this before running tests for this 10 goals to compare more or less success.
Of cause we know from the lecture Reinforcement Learning there are much better approaches like deep q-learning and similar approaches. 