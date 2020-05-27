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
