# -*- coding: utf-8 -*-

# input libraries
from RL.input import input_data
from RL.agent import agent
from RL.environment import environment
from RL.settings import parameters

# load object grid_data and run method import data frames
grid_data = input_data.InputData()
grid_data.import_csv_as_df()

# load object parameters
params = parameters.RLParameters()

# load object environment and initialize the case
env = environment.RLEnvironment(params, grid_data)
env.load_edges()
env.initialize_env_matrix()
env.set_cable_pieces()
env.set_number_of_states()
env.determine_max_cable_length()
env.set_unconnected_houses()

# load agent object and initialize variables
agent = agent.RLAgent(params, env)
agent.initialize_free_cable_pieces()