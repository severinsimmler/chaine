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