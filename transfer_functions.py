def network_param_difference(target_net, source_net):
    '''Calculates difference between two networks' parameters for biased regularization'''

    for param_target, param_source in zip(target_net.named_parameters(), source_net.named_parameters()):

        try:
            regularization_vector = param_target[1] - param_source[1]
        except:
            regularization_vector = param_target[1]
            print("Parameter mismatch in transfer learning. Likely incomatible dimensions in the last layer.")

        yield param_target[0], regularization_vector
