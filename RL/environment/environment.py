# -*- coding: utf-8 -*-

# input libraries
import numpy as np
import networkx as nx


class RLEnvironment:

    """
    Class defining the environment of the RL algorithm. Contains methods like
    - initializing the game board, i.e. environment
    - checking is game is over
    - resetting the environment
    """

    def __init__(self, settings, grid):
        self.grid = grid
        self.settings = settings
        self.env_matrix = None
        self.edges = None
        self.cable_pieces = None
        self.max_cable_length = 0
        self.number_states = 0
        self.unconnected_houses = 0
        self.reward = 0

    # initialize matrix with grid information
    """
    - all cable pieces
    - start and end nodes
    - chosen y/n
    - houses
    - length
    - moffen
    """
    def initialize_env_matrix(self):
        start_nodes = []
        end_nodes = []
        edges = []
        n_houses = []
        cable_lengths = []
        moffen = []
        level = []
        chosen = []
        for l in range(self.settings.cables):
            for index, row in self.edges.iterrows():
                start_nodes.append(row['START_NODE'])
                end_nodes.append(row['END_NODE'])
                edges.append(row['EDGE_ID'])
                n_houses.append(row['N_HOUSES'])
                cable_lengths.append(row['LENGTH'])
                moffen.append(0)
                level.append(l)
                chosen.append(0)

        self.env_matrix = np.asmatrix(np.column_stack([
            start_nodes,
            end_nodes,
            edges,
            n_houses,
            cable_lengths,
            level,
            chosen
        ]))

    # load the edges
    def load_edges(self):
        self.edges = self.grid.df_edges

    # set the vector with cable pieces
    def set_cable_pieces(self,):
        # number of cable pieces depend on number of cables specified in settings
        self.cable_pieces = np.arange(0, self.settings.cables * self.edges.shape[0])

    # set number of possible state
    def set_number_of_states(self):
        # each state may be 0 (empty) or 1 (chosen)
        self.number_states = 2 ** len(self.cable_pieces)

    # determine maximum cable length
    def determine_max_cable_length(self):
        # maximum cable length is length of all edges time the number of cables
        self.max_cable_length = self.edges['LENGTH'].sum() * self.settings.cables

    # set number of unconnected houses which will be used to determine if the agent is finished
    def set_unconnected_houses(self):
        self.unconnected_houses = self.edges['N_HOUSES'].sum()

    # update number of unconnected houses
    def update_unconnected_houses(self, edge):
        n_houses = self.env_matrix[edge, 3]
        self.unconnected_houses -= n_houses

    # determine state
    # there are 2 ** (number of edges * number of cables) states which all get their unique state id
    def get_state_id(self):
        enumerator = 0
        state = 0
        for i in range(len(self.cable_pieces)):
            if self.env_matrix[i, 6] == 0:  # i.e. if this cable piece is not chosen yet
                chosen = 0
            else:
                chosen = 1
            state += 2 ** enumerator * chosen
            enumerator += 1
        return state

    # check if the grid is complete, if all houses are connected and in the graph is connected
    def grid_finished(self):
        path_existence = []
        for l in range(self.settings.cables):
            path = self.verify_path_existence_level(self.env_matrix, self.grid.df_nodes, l)
            path_existence.append(path)

        if sum(path_existence) == self.settings.cables and self.unconnected_houses == 0:
            grid_finished = 1
        else:
            grid_finished = 0
        return grid_finished

    # calculate reward
    # reward only given when agent is finished and depends on cable length and number of moffen
    def get_reward(self):
        # r = n_moffen * cost_moffen + used_cable_length / max_cable_length * cost_cable_length
        n_moffen = self.determine_number_of_moffen(self.env_matrix, self.grid.df_nodes)
        trace_length = self.env_matrix[np.where(self.env_matrix[:, 6] == 1)[0], 4].sum()
        cost_moffen = n_moffen * self.settings.reward_mof
        cost_length = trace_length / self.max_cable_length * self.settings.reward_length
        self.reward = cost_moffen + cost_length

    # two helper functions are needed, one to determine whether the agent needs a mof when he chooses a cable piece and
    # another to check if the agent has a connected graph. If not he cannot be finished.
    @staticmethod
    def determine_number_of_moffen(env_matrix, df_nodes):
        g = nx.Graph()
        # add MSR node
        node_msr = df_nodes.loc[df_nodes['MSR'] > 0, 'NODE_ID'].values
        g.add_nodes_from(node_msr)

        # add nodes and edges from environment. These are the unique start and end nodes from the chosen cable pieces
        env_nodes = []
        env_edges = []
        for i in range(env_matrix.shape[0]):
            if env_matrix[i, 6] == 1:
                env_nodes = env_nodes + [env_matrix[i, 0], env_matrix[i, 1]]
                env_edges = env_edges + [(env_matrix[i, 0], env_matrix[i, 1])]
        # make unique
        env_nodes = list(set(env_nodes))
        env_edges = list(set(env_edges))

        # add to Graph
        g.add_nodes_from(env_nodes)
        g.add_edges_from(env_edges)

        # count all nodes that have more than 2 neighbours
        moffen = 0
        for node in g.nodes:
            neighbours = [n for n in g.neighbors(node)]
            if len(neighbours) > 2:
                moffen += 1
        return moffen

    @staticmethod
    # Determine if there exists a path from all edges to the MSR
    # This constraint holds for each level, i.e. each cable. There we check this as follows:
    # 1. Draw Networkx Graph for each level
    # 2. For all nodes in the Graph check if there exists a path from that node to the MSR
    def verify_path_existence_level(env_matrix, df_nodes, level):
        g = nx.Graph()
        # add MSR node
        node_msr = df_nodes.loc[df_nodes['MSR'] > 0, 'NODE_ID'].values
        g.add_nodes_from(node_msr)

        # add nodes and edges from environment. These are the unique start and end nodes from the chosen cable pieces
        env_nodes_lvl = []
        env_edges_lvl = []
        for i in range(env_matrix.shape[0]):
            if env_matrix[i, 5] == level and env_matrix[i, 6] == 1:
                env_nodes_lvl = env_nodes_lvl + [env_matrix[i, 0], env_matrix[i, 1]]
                env_edges_lvl = env_edges_lvl + [(env_matrix[i, 0], env_matrix[i, 1])]
        # make unique
        env_nodes_lvl = list(set(env_nodes_lvl))
        env_edges_lvl = list(set(env_edges_lvl))

        # add to Graph
        g.add_nodes_from(env_nodes_lvl)
        g.add_edges_from(env_edges_lvl)

        # check if graph is connected
        return nx.is_connected(g)
