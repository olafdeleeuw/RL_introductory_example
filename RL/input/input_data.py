# -*- coding: utf-8 -*-

# input libraries
import pandas as pd
import numpy as np
import networkx as nx
import logging

log = logging.getLogger(__name__)


class InputData:

    """
    class to import the example csv files with nodes and edges for the network and create Pandas data frames and
    dictionaries for plotting purposes.
    """

    def __init__(self):
        self.data_folder = "data/"
        self.filename_nodes = "nodes_example"
        self.filename_edges = "edges_example"
        self.df_edges = pd.DataFrame()
        self.df_nodes = pd.DataFrame()
        self.node_list = []
        self.node_list_msr = []
        self.node_list_households = []
        self.nodes_pos = {}
        self.cable1_pos = {}
        self.cable2_pos = {}
        self.household_nodes_pos = {}
        self.edge_tuples = []
        self.network_graph = nx.Graph()

    def update_filename_nodes(self, filename_new):
        self.filename_nodes = filename_new

    def update_filename_edges(self, filename_new):
        self.filename_edges = filename_new

    def import_csv_as_df(self):
        # load csv files with nodes and edges
        self.df_nodes = pd.read_csv(self.data_folder + self.filename_nodes + ".csv", sep=';')
        self.df_edges = pd.read_csv(self.data_folder + self.filename_edges + ".csv", sep=';')

    def create_node_list_without_msr(self):
        # create a list with all nodes except the MSR node to use in the networkx graph
        try:
            self.node_list = self.df_nodes.loc[self.df_nodes['MSR'] == 0, 'NODE_ID'].values.tolist()
        except ValueError as e:
            self.node_list = []
            log.warning("Column not found or dataframe is empty", e)

    def create_node_position_dict(self):
        pos = {}
        for index, row in self.df_nodes.iterrows():
            pos[index] = (row['NODE_X'], row['NODE_Y'])
        self.nodes_pos = pos

    def create_node_list_msr(self):
        # create a list with all MSR nodes to use in the networkx graph
        try:
            self.node_list_msr = self.df_nodes.loc[self.df_nodes['MSR'] == 1, 'NODE_ID'].values.tolist()
        except ValueError as e:
            self.node_list_msr = []
            log.warning("Column not found or dataframe is empty", e)

    def create_edge_tuples(self):
        # create a list with all edge tuples to use in the networkx graph
        edge_tuples = []
        try:
            for index, row in self.df_edges.iterrows():
                edge_tuples.append((row['START_NODE'], row['END_NODE']))
            self.edge_tuples = edge_tuples
        except ValueError as e:
            self.edge_tuples = []
            log.warning("Column not found or dataframe is empty", e)

    def create_household_nodes(self, number_of_houses=9):
        # create a list with household nodes.
        # Note this list is fixed for this example
        self.node_list_households = range(number_of_houses)
        self.household_nodes_pos = {0: (-50, 45),
                                    1: (-45, 45),
                                    2: (-40, 45),
                                    3: (-35, 45),
                                    4: (-30, 45),
                                    5: (15, 45),
                                    6: (20, 45),
                                    7: (55, 45),
                                    8: (60, 45)}

    def create_cable1_pos(self):
        self.cable1_pos = {8: (1, 0),
                           9: (1, 21),
                           10: (-119, 21),
                           11: (-119, 51),
                           12: (1, 51),
                           13: (101, 51),
                           14: (101, 21)}

    def create_cable2_pos(self):
        self.cable2_pos = {14: (2, 0),
                           15: (2, 22),
                           16: (-118, 22),
                           17: (-118, 52),
                           18: (2, 52),
                           19: (102, 52),
                           20: (102, 22)}

    def prepare_grid_data(self):
        # run the functions that create node and edges lists
        self.create_node_list_without_msr()
        self.create_node_list_msr()
        self.create_edge_tuples()
        self.create_node_position_dict()
        self.create_household_nodes()

    def update_networkx_graph(self, node_color='r', msr_color='b', house_color='g',
                              node_size=100, msr_size=150, house_size=50,
                              edge_color='b', edge_size=10):
        # add regular nodes to graph
        self.add_nodes_to_graph(self.network_graph, self.node_list, size=node_size, color=node_color)
        # add msr nodes to graph
        self.add_nodes_to_graph(self.network_graph, self.node_list_msr, size=msr_size, color=msr_color, label="MSR")
        # add household nodes to graph
        self.add_nodes_to_graph(self.network_graph, self.node_list_households, size=house_size, color=house_color)
        # add edges to graph
        self.add_edges_to_graph(self.network_graph, self.edge_tuples, size=edge_size, color=edge_color)

    @staticmethod
    def add_nodes_to_graph(graph, node_list, size=100, color='r', label=""):
        # add nodes and edges to the networkx graph
        graph.add_nodes_from(node_list, size=size, color=color, label=label)
        return graph

    @staticmethod
    def add_edges_to_graph(graph, edge_list, size=10, color='b'):
        # add nodes and edges to the networkx graph
        graph.add_edges_from(edge_list, size=size, color=color)
        return graph

    @staticmethod
    def create_nodes_and_edges_result(result):
        x1 = result[np.where(result[0:8, 6] == 1)[0], 0] + 8
        nodes_x1 = x1.reshape(x1.shape[0], ).tolist()[0]
        y1 = result[np.where(result[0:8, 6] == 1)[0], 1] + 8
        nodes_y1 = y1.reshape(y1.shape[0], ).tolist()[0]
        x2 = result[np.where(result[8:16, 6] == 1)[0], 0] + 14
        nodes_x2 = x2.reshape(x2.shape[0], ).tolist()[0]
        y2 = result[np.where(result[8:16, 6] == 1)[0], 1] + 14
        nodes_y2 = y2.reshape(y2.shape[0], ).tolist()[0]

        edge_tuples1 = []
        for i in range(len(nodes_x1)):
            edge1 = (nodes_x1[i], nodes_y1[i])
            edge_tuples1.append(edge1)
        edge_tuples2 = []
        for i in range(len(nodes_x2)):
            edge2 = (nodes_x2[i], nodes_y2[i])
            edge_tuples2.append(edge2)

        return edge_tuples1, edge_tuples2
