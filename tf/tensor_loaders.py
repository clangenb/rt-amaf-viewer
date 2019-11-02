######################################################
# Functions that load tensors or placeholders in order
# to explicitly reference them, when a trained model
# is loaded.
#######################################################


def load_xy(graph):
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')

    return x, y


def load_init_states(namescope, graph):
    return graph.get_tensor_by_name('{}/init_state/init_state:0'.format(namescope))


def load_current_states(namescope, graph):
    return graph.get_tensor_by_name('{}/lstm_network/current_state:0'.format(namescope))


def load_predictions(namescope, graph):
    return graph.get_tensor_by_name('{}/predictions/pred:0'.format(namescope))


def load_means_std(graph):
    feature_means = graph.get_tensor_by_name('feature_means_std/means:0')
    feature_std = graph.get_tensor_by_name('feature_means_std/std:0')

    return feature_means, feature_std


def load_mode(graph):
    return graph.get_tensor_by_name('mode:0')
