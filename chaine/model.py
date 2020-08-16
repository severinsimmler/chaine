from chaine.utils import TrainingMetadata
from chaine.data import Parameters
from dataclasses import dataclass
METADATA = TrainingMetadata()


class ConditionalRandomField:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    def train(self, dataset, optimizer, features):
        """Estimate parameters using the L-BFGS-B algorithm.

        Parameters
        ----------

        Returns
        -------
        """

        self.index2label = None
        self.num_labels = None


        self.feature_set = FeatureSet()
        self.feature_set.scan(self.training_data)
        self.index2label, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)

                
        params, log_likelihood, information = \
                fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,
                              x0=np.zeros(len(self.feature_set)),
                              args=(self.training_data, self.feature_set, training_feature_data,
                                    self.feature_set.get_empirical_counts(),
                                    self.index2label, self.squared_sigma),
                              callback=_callback)

    def predict(self, sentence):
        potential_table = _generate_potential_table(self.params, self.num_labels,
                                                    self.feature_set, X, inference=True)

        num_tokens = len(sentence)
        return self.viterbi(num_tokens, table)



class Tables:
    def __init__(self, sentence, inference: bool = True):
        self.inference = inference
        self._tables = list()

    def __getitem__(self, index: int):
        return self._tables[index]

    def _generate_potential_table(sentence, feature_set, inference=True):
        tables = list()
        for t in range(len(sentence)):
            table = np.zeros((self.num_labels, self.num_labels))
            if inference:
                for (prev_y, y), score in feature_set.calc_inner_products(params, X, t):
                    if prev_y == -1:
                        table[:, y] += score
                    else:
                        table[prev_y, y] += score
            else:
                for (prev_y, y), feature_ids in sentence[t]:
                    score = sum(params[fid] for fid in feature_ids)
                    if prev_y == -1:
                        table[:, y] += score
                    else:
                        table[prev_y, y] += score
            table = np.exp(table)
            if t == 0:
                table[1:] = 0
            else:
                table[:,0] = 0
                table[0,:] = 0
            tables.append(table)

        return tables