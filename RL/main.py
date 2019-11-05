# -*- coding: utf-8 -*-

# input libraries
import numpy as np
from RL.execution import execution as rl_exec


# Train the algorithm
for t in range(rl_exec.params.episodes):

    # run train iteration
    rl_exec.train_grid_planning()

    # reset
    if t % 500 == 0:
        print("episode: " + str(t))
        print("agents reward: " + str(rl_exec.env.reward))
        print("cable length: " + str(rl_exec.env.env_matrix[np.where(rl_exec.env.env_matrix[:, 6] == 1)[0], 4].sum()))
        print("number of moffen used: " + str(rl_exec.env.determine_number_of_moffen(rl_exec.env.env_matrix,
                                                                                     rl_exec.env.grid.df_nodes)))
        print("number of cables used: " + str(rl_exec.env.cables_used))

    # reset object for new run
    rl_exec.reset_env_elements()


# execute the algorithm with final results
rl_exec.execute_grid_planning()
