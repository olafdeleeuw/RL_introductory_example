# -*- coding: utf-8 -*-

# input libraries
import numpy as np


class RLAgent:

    """
    Class defining all actions and rewards regarding the agent
    - take action
    - get reward
    - state history
    - state values
    """

    def __init__(self, settings, rl_environment):
        self.settings = settings
        self.env = rl_environment
        self.alpha = 0.5  # learning rate
        self.free_cable_pieces = []
        self.state_value = []
        self.state_history = []
        self.action_history = []
        self.action = None
        self.state = None

    # initialize state values
    # first run: all states same value 0. No cable piece is better than another
    def initialize_state_values(self):
        # self.state_value = np.zeros(self.env.number_states)
        self.state_value = np.repeat([0.], self.env.number_states)

    # set all free cable pieces
    def initialize_free_cable_pieces(self):
        self.free_cable_pieces = self.env.cable_pieces

    # update free cable pieces
    def update_free_cable_pieces(self):
        self.free_cable_pieces = np.delete(self.free_cable_pieces, np.where(self.free_cable_pieces == self.action))

    # reset state history
    def reset_state_history(self):
        self.state_history = []

    # reset action history
    def reset_action_history(self):
        self.action_history = []

    # take action
    # follow epsilon-greedy policy
    def take_action(self):
        # pick random real number via uniform distribution
        r = np.random.rand()
        if r < self.settings.epsilon:
            # take random action, i.e. pick a random cable piece which is not chosen yet
            agents_choice = np.random.choice(self.free_cable_pieces)
            self.env.env_matrix[agents_choice, 6] = 1  # set potential cable piece as chosen
            state = self.env.get_state_id()  # get state if chosen
            self.env.env_matrix[agents_choice, 6] = 0  # change back
        else:
            # choose the cable piece with highest state value after the move
            # loop over all possible choices and check the state value after this move
            agents_choice = None
            state = None
            highest_state_value = min(self.state_value) - 1
            for f in self.free_cable_pieces:
                self.env.env_matrix[f, 6] = 1  # set potential cable piece as chosen
                potential_state = self.env.get_state_id()  # get state if chosen
                self.env.env_matrix[f, 6] = 0  # change back
                if self.state_value[potential_state] > highest_state_value:
                    highest_state_value = self.state_value[potential_state]
                    agents_choice = f
                    state = potential_state

        self.action = agents_choice
        self.state = state

    # update env
    def update_matrix_env(self):
        # update indicator if cable piece is chosen
        self.env.env_matrix[self.action, 6] = 1
        # update number of houses for and edge at each cable in case houses are connected. You cannot connect a house
        # twice
        self.env.update_unconnected_houses(self.action)
        cables_per_section = self.env.env_matrix.shape[0] / self.settings.cables
        self.env.env_matrix[np.where(self.env.env_matrix[:, 2] == (self.action % cables_per_section))[0], 3] = 0
        # update number of cables used
        cables = self.env.env_matrix[np.where(self.env.env_matrix[:, 6] == 1)[0], 5]
        self.env.cables_used = list(set(cables.reshape(cables.shape[0]).tolist()[0]))

    # update the action history
    def update_agents_state_history(self):
        self.state_history.append(self.state)
        self.action_history.append(self.action)

    # update state value
    # based on sort of backpropagation executed and end of an episode
    # V(prev_state) = V(prev_state) + alpha * (V(next_state) - V(prev_state))
    def update_state_value(self):
        # get reward
        reward = self.env.reward
        target_value = reward
        for prev_state in reversed(self.state_history):
            state_value = self.state_value[prev_state] + self.alpha * (target_value - self.state_value[prev_state])
            self.state_value[prev_state] = state_value
            target_value = state_value
