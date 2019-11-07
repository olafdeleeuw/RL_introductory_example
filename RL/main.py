# -*- coding: utf-8 -*-

# input libraries
import numpy as np
from RL.execution import execution as rl_exec


# Train the algorithm
for t in range(rl_exec.params.episodes):

    # run train iteration
    rl_exec.train_grid_planning()

    # reset
    if t % 1000 == 0:
        print("episode: " + str(t))
        print("agents reward: " + str(rl_exec.env.reward))
        print("cable length: " + str(rl_exec.env.env_matrix[np.where(rl_exec.env.env_matrix[:, 6] == 1)[0], 4].sum()))
        n_moffen = 0
        for l in rl_exec.env.cables_used:
            n_moffen += rl_exec.env.determine_number_of_moffen(rl_exec.env.env_matrix, rl_exec.env.grid.df_nodes, l)
        print("number of moffen used: " + str(n_moffen))
        print("cables used: " + str(rl_exec.env.cables_used))
        print("matrix: " + str(rl_exec.env.env_matrix))
        print("max state value: " + str(max(rl_exec.agent.state_value)))

    # reset object for new run
    rl_exec.reset_env_elements()


# execute the algorithm with final results
rl_exec.execute_grid_planning()
