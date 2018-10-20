import tensorflow as tf
import sonnet as snt

import graph_nets as gn
from graph_nets import utils_tf

class PGM(snt.AbstractModule):
    def __init__(self,
                 model_fn,
                 name="pgm"):
        """Initializes a PGM module.
        Args:
          model_fn: A callable, sigbature is, fn: edges x sender_nodes -> edge_vals
          name: The module name.
        """
        super(PGM, self).__init__(name=name)
        # TODO extend to hypergraphs (not just binary relations)
        with self._enter_variable_scope():
            self._model_fn = model_fn()

    def _build(self, graph):
        # gather from senders
        sender_nodes = gn.blocks.broadcast_sender_nodes_to_edges(graph)

        # apply edge to each relevant node
        node_inputs = self._model_fn(graph.edges, sender_nodes)

        # aggregate according to the receivers
        nodes = tf.unsorted_segment_sum(node_inputs, graph.receivers, tf.reduce_sum(graph.n_node))
        return graph.replace(nodes=tf.nn.selu(nodes))

def linear_model_fn(edges, nodes):
    """
    Treats the edges as linear functions and applies
    them to their respective sender nodes.

    Args:
        edges (tf.tensor): the edges of shape [n_edges, node_dim, node_dim]
        nodes (tf.tensor): the sender nodes of the n_edges. [n_edges, node_dim]

    Returns:
        (tf.tensor): tensor of shape [n_edges, node_dim]
    """
    # transform(signal) = edge(node) = edge x node
    return tf.squeeze(tf.matmul(edges, tf.expand_dims(nodes, -1)))
