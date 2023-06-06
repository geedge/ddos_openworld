import tensorflow as tf

class SAGE(tf.keras.Model):
    """
    SAGE layer class.
    """
    def __init__(self, args, number_of_features):
        """
        Creating a SAGE layer.
        :param args: Arguments object.
        :param number_of_features: Number of node features.
        """
        super(SAGE, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self._setup()

    def _setup(self):
        """
        Setting up upstream and pooling layers.
        """
        self.graph_convolution_1 = tf.keras.layers.Dense(units=self.args.first_gcn_dimensions,
                                                         activation='relu',
                                                         input_shape=(self.number_of_features,))
        self.graph_convolution_2 = tf.keras.layers.Dense(units=self.args.second_gcn_dimensions,
                                                         activation='relu')
        self.fully_connected_1 = tf.keras.layers.Dense(units=self.args.first_dense_neurons,
                                                       activation='tanh')
        self.fully_connected_2 = tf.keras.layers.Dense(units=self.args.second_dense_neurons,
                                                       activation='softmax')

    def call(self, data):
        """
        Making a forward pass with the graph level data.
        :param data: Data feed dictionary.
        :return graph_embedding: Graph level embedding.
        :return penalty: Regularization loss.
        """
        edges = tf.convert_to_tensor(data["edge"])
        features = tf.convert_to_tensor(data["features"])
        node_features_1 = tf.nn.relu(self.graph_convolution_1(features))
        node_features_2 = tf.nn.relu(self.graph_convolution_2(node_features_1))
        abstract_features_1 = tf.math.tanh(self.fully_connected_1(node_features_2))
        attention = tf.nn.softmax(self.fully_connected_2(abstract_features_1), axis=0)
        attention = tf.transpose(attention)
        graph_embedding = tf.matmul(tf.transpose(attention), node_features_2)
        graph_embedding = tf.reshape(graph_embedding, [1, -1])
        penalty = tf.matmul(tf.transpose(attention), attention) - tf.eye(self.args.second_dense_neurons)
        penalty = tf.norm(penalty, ord=2)
        return graph_embedding, penalty
