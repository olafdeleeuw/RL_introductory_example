# -*- coding: utf-8 -*-

# input libraries
import numpy as np
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
agent.initialize_state_values()


# execute grid planning
def train_grid_planning():
    while not env.grid_finished():
        agent.take_action()
        # print("agents action: " + str(agent.action))
        agent.update_matrix_env()
        # print(env.env_matrix)
        agent.update_agents_state_history()
        agent.update_free_cable_pieces()

        # print("free cables: " + str(agent.free_cable_pieces))
        # print("state history: " + str(agent.state_history))
        # print("cables used: " + str(env.cables_used))
        # print("unconnected houses: " + str(env.unconnected_houses))
        # print("min state value: " + str(min(agent.state_value)))
        # print("finished......: " + str(env.grid_finished()))

    env.get_reward()
    agent.update_state_value()


# reset environment settings and objects
def reset_env_elements():
    agent.reset_state_history()
    agent.initialize_free_cable_pieces()
    env.initialize_env_matrix()
    env.set_unconnected_houses()
    env.reset_cables_used()


def execute_grid_planning():
    # set epsilon to zero
    params.epsilon = 0
    while not env.grid_finished():
        agent.take_action()
        # print("agents action: " + str(agent.action))
        agent.update_matrix_env()
        agent.update_free_cable_pieces()

    print("cable length: " + str(env.env_matrix[np.where(env.env_matrix[:, 6] == 1)[0], 4].sum()))
    print("number of moffen used: " + str(env.determine_number_of_moffen(env.env_matrix, env.grid.df_nodes)))
    print("number of cables used: " + str(env.cables_used))
    print("env matrix: " + str(env.env_matrix))
