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
def execute_grid_planning():
    while not env.grid_finished():
        agent.take_action()
        agent.update_matrix_env()
        agent.update_agents_action_history()
        agent.update_free_cable_pieces()

        # print(agent.free_cable_pieces)
        # print(agent.action_history)
        # print(env.unconnected_houses)

        print("number of moffen: " + str(env.env_matrix[:, 5].sum()))
        print("cable length: " + str(env.env_matrix[np.where(env.env_matrix[:, 7] == 1)[0], 4].sum()))

    env.get_reward()
    agent.update_state_value()
    print(env.reward)


for t in range(params.episodes):
    if t % 200 == 0:
        print(t)
    execute_grid_planning()



