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
        self.action_history = []
        self.action = None

    # initialize state values
    # first run: all states same value 0. No cable piece is better than another
    def initialize_state_values(self):
        self.state_value = np.repeat([0], self.env.number_states)

    # set all free cable pieces
    def initialize_free_cable_pieces(self):
        self.free_cable_pieces = self.env.cable_pieces

    # update free cable pieces
    def update_free_cable_pieces(self, agents_choice):
        self.free_cable_pieces.remove(agents_choice)

    # reset state history
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
        else:
            # choose the cable piece with highest state value after the move
            # loop over all possible choices and check the state value after this move
            agents_choice = None
            highest_state_value = 0
            for f in self.free_cable_pieces:
                self.env.env_matrix[f, 7] = 1  # set potential cable piece as chosen
                potential_state = self.env.get_state()  # get state if chosen
                self.env.env_matrix[f, 7] = 0  # change back
                if self.state_value[potential_state] > highest_state_value:
                    highest_state_value = self.state_value[potential_state]
                    agents_choice = f

        self.action = agents_choice

    # update env
    def update_matrix_env(self):
        # update indicator if cable piece is chosen
        self.env.env_matrix[self.action, 7] = 1
        # update if a mof is needed
        self.env.env_matrix[self.action, 5] = self.determine_mof_needed_for_edge(self.env.env_matrix, self.action)

    # update the action history
    def update_agents_action_history(self):
        self.action_history.append(self.action)

    # update state value
    # based on sort of backpropagation executed and end of an episode
    # V(prev_state) = V(prev_state) + alpha * (V(next_state) - V(prev_state))
    def update_state_value(self):
        # get reward
        reward = self.env.reward
        target_value = reward
        for prev_state in reversed(self.action_history):
            state_value = self.state_value[prev_state] + self.alpha * (target_value - self.state_value[prev_state])
            self.state_value[prev_state] = state_value
            target_value = state_value

    # two helper functions are needed, one to determine whether the agent needs a mof when he chooses a cable piece and
    # another to check if the agent has a connected graph. If not he cannot be finished.
    @staticmethod
    def determine_mof_needed_for_edge(env_matrix, agents_choice):
        selected_edge = env_matrix[agents_choice, :]

        # select all items from the same cable which are already chosen by agent
        lvl_chosen_items = env_matrix[np.where(env_matrix[:, 6] == selected_edge[:, 6])[0], :]
        lvl_chosen_items = lvl_chosen_items[np.where(lvl_chosen_items[:, 7] == 1)[0], :]

        # check if the start and end node from the selected edge already appear twice
        selected_start = selected_edge[0, 0]
        selected_end = selected_edge[0, 1]
        chosen_start_items1 = lvl_chosen_items[np.where(lvl_chosen_items[:, 0] == selected_start)[0], :]
        chosen_start_items2 = lvl_chosen_items[np.where(lvl_chosen_items[:, 1] == selected_start)[0], :]
        chosen_start_items = np.concatenate((chosen_start_items1, chosen_start_items2), axis=0)

        chosen_end_items1 = lvl_chosen_items[np.where(lvl_chosen_items[:, 0] == selected_end)[0], :]
        chosen_end_items2 = lvl_chosen_items[np.where(lvl_chosen_items[:, 1] == selected_end)[0], :]
        chosen_end_items = np.concatenate((chosen_end_items1, chosen_end_items2), axis=0)

        # determine if a mof is needed
        if chosen_start_items.shape[0] > 1 or chosen_end_items.shape[0] > 1:
            mof = 1
        else:
            mof = 0

        return mof
