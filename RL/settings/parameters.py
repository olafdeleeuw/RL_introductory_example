# -*- coding: utf-8 -*-


class RLParameters:

    """
    class with basic model parameters.
    - number of episodes
    - values of rewards
    """

    def __init__(self):
        self.reward_length = -2
        self.reward_mof = -3
        self.reward_all_cables_used = .5
        self.episodes = 200000
        self.cables = 2
        self.learning_rate = 0.5
        self.epsilon = 0.1

    def set_number_of_episodes(self, episodes):
        self.episodes = episodes

    def set_number_of_cables(self, cables):
        self.cables = cables

    def update_reward_length(self, new_reward_length):
        self.reward_length = new_reward_length

    def update_reward_mof(self, new_reward_mof):
        self.reward_mof = new_reward_mof

    def update_reward_cables_uses(self, new_reward_cables_used):
        self.reward_all_cables_used = new_reward_cables_used

    # def update_reward_houses(self, new_reward_houses):
    #     self.reward_connected_houses = new_reward_houses

    def update_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate

    def update_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
