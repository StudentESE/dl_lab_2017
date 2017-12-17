from network import Network
from utils   import Options, rgb2gray
from random import randrange

opt = Options()

class Agent:

    def __init__(self):
        """
        Intitialize the agent, i.e. its neural network and history.
        """
        self.network = Network(opt.device, opt.path)
        self.history = []


    def train(self, training_data, valid_data):
        """
        Train the agents neural network with given training_data.

        Hyperparameters of training (num_epochs, learning_rate, batch_size)
        can be adjusted in the Options class.

        Returns accuracy on valid_data
        """
        self.network.train(training_data, valid_data, opt.num_epochs, opt.learning_rate, opt.minibatch_size)

        # Compute accuracy on valid_data
        return self.network.compute_accuracy(valid_data)

    def compute_next_action(self, state):
        """
        Computes the next action, given the current state and the agents history.

        If the history is less than opt.hist_len, the agent first performs
        some random actions in order to explore its neighbourhood.

        Returns action (i.e. value in [0, opt.act_num])
        """
        # Compute gray scale image and append it to history
        gray_state = rgb2gray(state.pob).reshape(opt.state_siz)
        self.history.append(gray_state)

        # Check if we have enough history to predit action, else perform random action
        if len(self.history) > opt.hist_len:
            self.history.pop(0) # TODO: if neccesary, improve performance (collections.deqeue)

            # Create network input x from history
            x = []
            for s in self.history:
                for value in s:
                    x.append(value)
            i = self.network.predict(x)
            return i
        else:
            return randrange(opt.act_num)
