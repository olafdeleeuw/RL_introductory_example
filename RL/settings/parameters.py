# -*- coding: utf-8 -*-


class RLParameters:

    """
    class with basic model parameters.
    - number of episodes
    - values of rewards
    """

    def __init__(self):
        self.reward_length = 1
        self.reward_mof = 10
        self.reward_connected_houses = 25
        self.episodes = 1e4

    def set_number_of_episodes(self, episodes):
        self.episodes = episodes

    def update_reward_length(self, new_reward_length):
        self.reward_length = new_reward_length

    def update_reward_mof(self, new_reward_mof):
        self.reward_mof = new_reward_mof

    def update_reward_houses(self, new_reward_houses):
        self.reward_connected_houses = new_reward_houses
