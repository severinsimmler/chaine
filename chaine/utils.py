class TrainingMetadata:
    def __init__(self):
        self.iteration = 0
        self.sub_iteration = 0
        self.total_sub_iterations = 0
        self.gradient = None

    def __call__(self, args):
        self.iteration += 1
        self.total_sub_iterations += self.sub_iteration
        self.sub_iteration = 0


def calc_inner_products(self, params, X, t):
    # TODO for inference
    inner_products = Counter()
    for feature_string in self.feature_func(X, t):
        try:
            for (prev_y, y), feature_id in self.feature_dic[feature_string].items():
                inner_products[(prev_y, y)] += params[feature_id]
        except KeyError:
            pass
    return [((prev_y, y), score) for (prev_y, y), score in inner_products.items()]


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