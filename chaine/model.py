from chaine.utils import TrainingMetadata


METADATA = TrainingMetadata()



class ConditionalRandomField:
    def train(self, dataset, optimizer):
        """Estimate parameters using the L-BFGS-B algorithm.

        Parameters
        ----------

        Returns
        -------
        """

        self.index2label = None
        self.num_labels = None
        self.parameters = self.estimate_parameters()


        # Generate feature set from the corpus
        self.feature_set = FeatureSet()
        self.feature_set.scan(self.training_data)
        self.index2label, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        print("* Number of labels: %d" % (self.num_labels-1))
        print("* Number of features: %d" % len(self.feature_set))

        # Estimates parameters to maximize log-likelihood of the corpus.
        self._estimate_parameters()

        self.save_model(model_filename)

    def export(self):
        return {"feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.label_array,
                 "params": list(self.params)}


    def save(self, filepath):
        model = self.export()
        with Path(filepath).open("w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, model_filename):
        f = open(model_filename)
        model = json.load(f)
        f.close()

        self.feature_set = FeatureSet()
        self.feature_set.load(model['feature_dic'], model['num_features'], model['labels'])
        self.index2label, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        self.params = np.asarray(model['params'])

    def predict(self, sentence):
        potential_table = _generate_potential_table(self.params, self.num_labels,
                                                    self.feature_set, X, inference=True)

        num_tokens = len(sentence)
        return self.viterbi(num_tokens, table)


    def estimate_parameters(self):
        training_feature_data = self._get_training_feature_data()
        print('* Squared sigma:', self.squared_sigma)
        print('* Start L-BGFS')
        print('   ========================')
        print('   iter(sit): likelihood')
        print('   ------------------------')
        params, log_likelihood, information = \
                fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,
                              x0=np.zeros(len(self.feature_set)),
                              args=(self.training_data, self.feature_set, training_feature_data,
                                    self.feature_set.get_empirical_counts(),
                                    self.index2label, self.squared_sigma),
                              callback=_callback)
        print('   ========================')
        print('   (iter: iteration, sit: sub iteration)')
        print('* Training has been finished with %d iterations' % information['nit'])

        if information['warnflag'] != 0:
            print('* Warning (code: %d)' % information['warnflag'])
            if 'task' in information.keys():
                print('* Reason: %s' % (information['task']))
        print('* Likelihood: %s' % str(log_likelihood))
        return params


    def viterbi(self, num_tokens: int, potential_table):
        """Viterbi algorithm with backpointers.

        Finds the most likely sequence of hidden states that results in a sequence of observed events.

        Parameters
        ----------
        num_tokens : int
            bla
        table : bla
            bla

        Returns
        -------
        bla
            blabla
        """
        max_table = np.zeros((num_tokens, self.num_labels))
        argmax_table = np.zeros((num_tokens, self.num_labels), dtype='int64')

        t = 0
        for label_id in range(self.num_labels):
            max_table[t, label_id] = potential_table[t][0, label_id]
        for t in range(1, num_tokens):
            for label_id in range(1, self.num_labels):
                max_value = -float('inf')
                max_label_id = None
                for prev_label_id in range(1, self.num_labels):
                    value = max_table[t-1, prev_label_id] * potential_table[t][prev_label_id, label_id]
                    if value > max_value:
                        max_value = value
                        max_label_id = prev_label_id
                max_table[t, label_id] = max_value
                argmax_table[t, label_id] = max_label_id

        sequence = list()
        next_label = max_table[num_tokens-1].argmax()
        sequence.append(next_label)
        for t in range(num_tokens-1, -1, -1):
            next_label = argmax_table[t, next_label]
            sequence.append(next_label)
        return [self.index2label[label_id] for label_id in sequence[::-1][1:]]



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