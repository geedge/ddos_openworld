"""SEAL-CI model."""

import random
from tqdm import trange
from utils import hierarchical_graph_reader, GraphDatasetGenerator
from sklearn import metrics
import pandas as pd
import numpy as np

class SEALCITrainer(object):
    """
    Semi-Supervised Graph Classification: A Hierarchical Graph Perspective Cautious Iteration model.
    """
    def __init__(self, args):
        """
        Creating dataset, doing dataset split, creating target and node index vectors.
        :param args: Arguments object.
        """
        self.args = args
        self.macro_graph = hierarchical_graph_reader(self.args.hierarchical_graph)  # 大图
        self.dataset_generator = GraphDatasetGenerator(self.args.graphs, self.args.feature_which)
        self._setup_macro_graph()  # 大图的边 边给加好了self.macro_graph_edges
        #self._create_split()  # 区分了带标签的和不带标签的 self.labled_indices, self.unlabeld_indices
        #self._create_labeled_target()  # self.labeled_mask, self.labeled_target
        self._create_node_indices()  # node_indices

    def _setup_macro_graph(self):
        """
        Creating an edge list for the hierarchical graph.
        """
        self.macro_graph_edges = [[edge[0], edge[1]] for edge in self.macro_graph.edges()]
        self.macro_graph_edges = self.macro_graph_edges + [[edge[1], edge[0]] for edge in self.macro_graph.edges()]
        self.macro_graph_edges = np.array(self.macro_graph_edges).T

    def _create_node_indices(self):
        """
        Creating an index of nodes.
        """
        self.node_indices = [index for index in range(self.macro_graph.number_of_nodes())]
        self.node_indices = np.array(self.node_indices)



