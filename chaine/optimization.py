import scipy.optimize
import numpy as np


class Optimizer:
    def __init__(self, dataset: Dataset, squared_sigma: float = 10.0):
        self.squared_sigma = squared_sigma
        self.dataset = dataset
        self.callback = Callback()

    def log_likelihood(self, parameters):
        expected_counts = np.zeros(len(self.dataset))

        total_log_z = 0
        for sentence in self.dataset:
            table = self.potential_table(parameters, inference=False)

            alpha, beta, z, scaling = self.forward_backward(table)
            total_log_z += math.log(z) + sum(
                math.log(coefficient) for coefficient in scaling.values()
            )

            for t in range(len(sentence)):
                potential = table[t]
                for (prev_y, y), feature_ids in X_features[t]:
                    if prev_y == -1:
                        if t in scaling.keys():
                            prob = (alpha[t, y] * beta[t, y] * scaling[t]) / Z
                        else:
                            prob = (alpha[t, y] * beta[t, y]) / Z
                    elif t == 0:
                        if prev_y is not STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y]) / Z
                    else:
                        if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                            continue
                        else:
                            prob = (alpha[t - 1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                    for fid in feature_ids:
                        expected_counts[fid] += prob

        likelihood = (
            np.dot(dataset.features.empirical_counts, parameters)
            - total_log_z
            - np.sum(np.dot(parameters, parameters)) / squared_sigma * 2
        )

        self.callback.gradient = (
            dataset.features.empirical_counts - expected_counts - parameters / squared_sigma
        )
        self.callback.sub_iteration += 1

        return likelihood * -1

    def optimize(self):
        x0 = np.zeros(len(self.dataset.features))
        return scipy.optimize.fmin_l_bfgs_b(
            func=self.log_likelihood, fprime=self.gradient, callback=self.callback, x0=x0,
        )

    def forward_backward(self, num_labels, time_length, potential_table):
        alpha = np.zeros((time_length, num_labels))
        scaling = dict()
        t = 0
        for label_id in range(num_labels):
            alpha[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]
        t = 1
        while t < time_length:
            scaling_time = None
            scaling_coefficient = None
            overflow_occured = False
            label_id = 1
            while label_id < num_labels:
                alpha[t, label_id] = np.dot(alpha[t - 1, :], potential_table[t][:, label_id])
                if alpha[t, label_id] > SCALING_THRESHOLD:
                    if overflow_occured:
                        print("******** Consecutive overflow ********")
                        raise BaseException()
                    overflow_occured = True
                    scaling_time = t - 1
                    scaling_coefficient = SCALING_THRESHOLD
                    scaling[scaling_time] = scaling_coefficient
                    break
                else:
                    label_id += 1
            if overflow_occured:
                alpha[t - 1] /= scaling_coefficient
                alpha[t] = 0
            else:
                t += 1

        beta = np.zeros((time_length, num_labels))
        t = time_length - 1
        for label_id in range(num_labels):
            beta[t, label_id] = 1.0
        for t in range(time_length - 2, -1, -1):
            for label_id in range(1, num_labels):
                beta[t, label_id] = np.dot(beta[t + 1, :], potential_table[t + 1][label_id, :])
            if t in scaling.keys():
                beta[t] /= scaling[t]

        z = sum(alpha[time_length - 1])

        return alpha, beta, z, scaling

    def _gradient(params, *args):
        return self.callback.gradient * -1
