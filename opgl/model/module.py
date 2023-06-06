# coding=utf-8

import tf_geometric as tfg
import tensorflow as tf

class SAGE(tf.keras.Model):
    """
    SAGE layer class.
    """
    def __init__(self, args):
        """
        Creating a SAGE layer.
        :param args: Arguments object.
        :param number_of_features: Number of node features.
        """
        super(SAGE, self).__init__()
        self.args = args
        # self.number_of_features = number_of_features
        self._setup()

    def _setup(self):
        """
        Setting up upstream and pooling layers.
        """
        self.graph_convolution_1 = tfg.layers.GCN(self.args.first_gcn_dimensions)
        self.graph_convolution_2 = tfg.layers.GCN(self.args.second_gcn_dimensions)
        self.fully_connected_1 = tf.keras.layers.Dense(self.args.first_dense_neurons)
        self.fully_connected_2 = tf.keras.layers.Dense(self.args.second_dense_neurons)

    def call(self, data):
        """
        Making a forward pass with the graph level data.
        :param data: Data feed dictionary.
        :return graph_embedding: Graph level embedding.
        :return penalty: Regularization loss.
        """
        features, edges, node_graph_index = data.x, data.edge_index, data.node_graph_index
        node_features_1 = tf.nn.relu(self.graph_convolution_1([features, edges]))
        node_features_2 = self.graph_convolution_2([node_features_1, edges])
        abstract_features_1 = tf.nn.tanh(self.fully_connected_1(node_features_2))
        attention = tf.nn.softmax(self.fully_connected_2(abstract_features_1), axis=0)
        # attention = tf.transpose(attention)
        embeddings = []
        penalties = 0
        for graph in range(len(data.graphs)):
            index = tf.where(data.node_graph_index == graph)
            attention_tmp = tf.gather_nd(attention, index)
            node_features_2_tmp = tf.gather_nd(node_features_2, index)
            graph_embedding_tmp = tf.matmul(tf.transpose(attention_tmp), node_features_2_tmp)
            graph_embedding_tmp = tf.reshape(graph_embedding_tmp, [1, -1])
            penalty_tmp = tf.matmul(tf.transpose(attention_tmp), attention_tmp) - tf.eye(self.args.second_dense_neurons)
            penalty_tmp = tf.reduce_sum(tf.norm(penalty_tmp, ord=2, axis=1))
            embeddings.append(graph_embedding_tmp)
            penalties += penalty_tmp
          # -》error 2.0
        embeddings = tf.concat(embeddings, 0)
        penalties = penalties/len(data.graphs)
        #graph_embedding = tf.matmul(tf.transpose(attention), node_features_2)
        #graph_embedding = tf.reshape(graph_embedding, [1, -1])
        #penalty = tf.matmul(tf.transpose(attention), attention) - tf.eye(self.args.second_dense_neurons)
        #penalty = tf.reduce_sum(tf.norm(penalty, ord=2, axis=1))
        return embeddings, penalties

class GoG(tf.keras.Model):
    def __init__(self, args, num_units_list, output_list=False, uncertain=True):
        super().__init__()
        """
        Creating a SEAl-CI layer.
        :param args: Arguments object.
        :param number_of_features: Number of features per graph.
        :param number_of_labels: Number of node level labels.
        """
        self.args = args
        self.num_units_list = num_units_list
        self.output_list = output_list
        self.uncertain = uncertain
        self._setup()

    def _setup(self):
        self.graph_level_model = SAGE(self.args) # -> 有问题
        self.hierarchical_model = MultiVariationalGCNWithDense(self.num_units_list, self.output_list, self.uncertain)

    def call(self, inputs, training=None, mask=None, cache=None):
        graphs, edges = inputs
        #for graph in graphs:
        embeddings, penalties = self.graph_level_model(graphs)  # -》error 2.0
        h, kl = self.hierarchical_model([embeddings, edges], training, mask, cache)
        return h, kl, penalties

class MultiVariationalGCN(tf.keras.Model):

    def __init__(self, num_units_list, output_list=False, uncertain=True):
        super().__init__()

        self.uncertain = uncertain
        self.output_list = output_list
        self.gcns = [
            tfg.layers.GCN(num_units, activation=tf.nn.relu)
            for i, num_units in enumerate(num_units_list[:-1])
        ]

        self.mu_gcn = tfg.layers.GCN(num_units_list[-1])
        if self.uncertain:
            self.std_gcn = tfg.layers.GCN(num_units_list[-1])


    def call(self, inputs, training=None, mask=None, cache=None):
        h, edge_index = inputs
        g = tfg.Graph(
            x=h,
            edge_index=edge_index,
            #y=graph_dict["graph_label"]  # graph_dict["graph_label"] is a list with one int element
        )
        h, edge_index = g.x, g.edge_index
        outputs = []


        for i, gcn in enumerate(self.gcns):
            h = gcn([h, edge_index], cache=cache)
            outputs.append(h)

        mu = self.mu_gcn([h, edge_index], cache=cache)
        if self.uncertain:
            log_std = self.std_gcn([h, edge_index], cache=cache)
            std = tf.math.exp(log_std)
            rand = tf.random.truncated_normal(mu.shape)
        # if training and False:
        if self.uncertain and training:
            h = mu + rand * std
            kl = -log_std + 0.5 * (std ** 2 + mu ** 2 - 1)
        else:
            h = mu
            kl = mu * 0.0
        outputs.append(h)


        if self.output_list:
            return outputs, kl
        else:
            return h, kl



class MultiVariationalGCNWithDense(MultiVariationalGCN):
    def __init__(self, num_units_list, output_list=False, uncertain=True):
        super().__init__(num_units_list[:-1], output_list=output_list, uncertain=uncertain)
        self.dense = tf.keras.layers.Dense(num_units_list[-1])

    def call(self, inputs, training=None, mask=None, cache=None):
        super_results, kl = super().call(inputs, training=training, mask=mask, cache=cache)
        if self.output_list:
            outputs = super_results
            h = outputs[-1]
        else:
            h = super_results

        # h = tf.nn.relu(h)
        h = self.dense(h)

        if self.output_list:
            outputs.append(h)
            return outputs, kl
        else:
            return h, kl

