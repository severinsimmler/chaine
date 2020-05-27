import scipy.optimize



class Optimizer:
    def __init__(self):
        pass

    def log_likelihood(self, parameters, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma):
        training_data, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma = args
        expected_counts = np.zeros(len(feature_set))

        total_logZ = 0
        for X_features in training_feature_data:
            potential_table = _generate_potential_table(parameters, len(label_dic), feature_set,
                                                        X_features, inference=False)
            alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
            total_logZ += log(Z) + \
                        sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
            for t in range(len(X_features)):
                potential = potential_table[t]
                for (prev_y, y), feature_ids in X_features[t]:
                    # Adds p(prev_y, y | X, t)
                    if prev_y == -1:
                        if t in scaling_dic.keys():
                            prob = (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
                        else:
                            prob = (alpha[t, y] * beta[t, y])/Z
                    elif t == 0:
                        if prev_y is not STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y])/Z
                    else:
                        if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (alpha[t-1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                    for fid in feature_ids:
                        expected_counts[fid] += prob

        likelihood = np.dot(empirical_counts, parameters) - total_logZ - \
                    np.sum(np.dot(parameters,parameters))/(squared_sigma*2)

        gradients = empirical_counts - expected_counts - parameters/squared_sigma
        callback.gradient = gradients

        sub_iteration_str = '    '
        if callback.sub_iteration > 0:
            sub_iteration_str = '(' + '{0:02d}'.format(999999) + ')'
        print('  ', '{0:03d}'.format(callback.iteration), sub_iteration_str, ':', likelihood * -1)

        callback.sub_iteration += 1

        return likelihood * -1



parameters, log_likelihood, information = scipy.optimize.fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,
                        x0=np.zeros(len(self.feature_set)),
                        args=(self.training_data, self.feature_set, training_feature_data,
                            self.feature_set.get_empirical_counts(),
                            self.label_dic, self.squared_sigma),
                        callback=callback)







def _forward_backward(num_labels, time_length, potential_table):
    """
    Calculates alpha(forward terms), beta(backward terms), and Z(instance-specific normalization factor)
        with a scaling method(suggested by Rabiner, 1989).
    * Reference:
        - 1989, Lawrence R. Rabiner, A Tutorial on Hidden Markov Models and Selected Applications
        in Speech Recognition
    """
    alpha = np.zeros((time_length, num_labels))
    scaling_dic = dict()
    t = 0
    for label_id in range(num_labels):
        alpha[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
    #alpha[0, :] = potential_table[0][STARTING_LABEL_INDEX, :]  # slow
    t = 1
    while t < time_length:
        scaling_time = None
        scaling_coefficient = None
        overflow_occured = False
        label_id = 1
        while label_id < num_labels:
            alpha[t, label_id] = np.dot(alpha[t-1,:], potential_table[t][:,label_id])
            if alpha[t, label_id] > SCALING_THRESHOLD:
                if overflow_occured:
                    print('******** Consecutive overflow ********')
                    raise BaseException()
                overflow_occured = True
                scaling_time = t - 1
                scaling_coefficient = SCALING_THRESHOLD
                scaling_dic[scaling_time] = scaling_coefficient
                break
            else:
                label_id += 1
        if overflow_occured:
            alpha[t-1] /= scaling_coefficient
            alpha[t] = 0
        else:
            t += 1

    beta = np.zeros((time_length, num_labels))
    t = time_length - 1
    for label_id in range(num_labels):
        beta[t, label_id] = 1.0
    #beta[time_length - 1, :] = 1.0     # slow
    for t in range(time_length-2, -1, -1):
        for label_id in range(1, num_labels):
            beta[t, label_id] = np.dot(beta[t+1,:], potential_table[t+1][label_id,:])
        if t in scaling_dic.keys():
            beta[t] /= scaling_dic[t]

    Z = sum(alpha[time_length-1])

    return alpha, beta, Z, scaling_dic


def _gradient(params, *args):
    return GRADIENT * -1
