# -*- coding: utf-8 -*-

# input libraries
import numpy as np


class RLEnvironment:

    """
    Class defining the environment of the RL algorithm. Contains methods like
    - initializing the game board, i.e. environment
    - checking is game is over
    - resetting the environment
    """

    def __init__(self, settings):
        self.settings = settings
        self.env_edges = None
        self.unconnected_houses = 0
        self.game_over = 0

    # Initialization of the environment
    def initialize_env_matrix(self, df_edges, max_cables=2):
        # matrix with 8 columns:
        # 0. start_node
        # 1. end_node
        # 2. edge
        # 3. n_houses
        # 4. cable_length
        # 5. mof
        # 6. level
        # 7. chosen
        #
        # n rows where n = max_cables * number of edges

        start_nodes = []
        end_nodes = []
        edges = []
        n_houses = []
        cable_lengths = []
        moffen = []
        level = []
        chosen = []
        for l in range(max_cables):
            for index, row in df_edges.iterrows():
                start_nodes.append(row['start_node'])
                end_nodes.append(row['end_node'])
                edges.append(row['edge'])
                n_houses.append(row['n_houses'])
                cable_lengths.append(row['cable_length'])
                moffen.append(0)
                level.append(l)
                chosen.append(0)

        self.env_edges = np.asmatrix(np.column_stack([
            start_nodes,
            end_nodes,
            edges,
            n_houses,
            cable_lengths,
            moffen,
            level,
            chosen
        ]))
