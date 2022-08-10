"""
chaine.training
~~~~~~~~~~~~~~~

This module implements the high-level API to train a conditional random field.
"""


from chaine.crf import HyperparameterOptimizer, Model, Trainer
from chaine.logging import Logger, set_verbosity
from chaine.typing import Filepath, Iterable, Labels, Sequence

LOGGER = Logger(__name__)


def train(
    dataset: Iterable[Sequence],
    labels: Iterable[Labels],
    *,
    model_filepath: Filepath = "model.chaine",
    optimize_hyperparameters: bool = False,
    optimization_sample_size: int | None = None,
    verbose: int = 1,
    **hyperparameters,
) -> Model:
    """Train a conditional random field.

    Parameters
    ----------
    dataset : Iterable[Sequence]
        Data set consisting of sequences of feature sets.
    labels : Iterable[Labels]
        Labels corresponding to each instance in the data set.
    model_filepath : Filepath, optional (default=model.chaine)
        Path to model location.
    optimize_hyperparameters : bool
        If True, optimize hyperparameters first.
    optimization_sample_size : int | None
        Number of instances to sample from the data set for hyperparameter optimization.
    verbose : int
        Controls the verbosity: the higher, the more messages.
    algorithm : str
        The following optimization algorithms are available:
            * lbfgs: Limited-memory BFGS with L1/L2 regularization
            * l2sgd: Stochastic gradient descent with L2 regularization
            * ap: Averaged perceptron
            * pa: Passive aggressive
            * arow: Adaptive regularization of weights

    Limited-memory BFGS Parameters (lbfgs)
    --------------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (unlimited by default).
    num_memories : int, optional (default=6)
        Number of limited memories for approximating the inverse hessian matrix.
    c1 : float, optional (default=0)
        Coefficient for L1 regularization.
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization.
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion.
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this.
    linesearch : str, optional (default="MoreThuente")
        Line search algorithm used in updates:
            * MoreThuente: More and Thuente's method
            * Backtracking: Backtracking method with regular Wolfe condition
            * StrongBacktracking: Backtracking method with strong Wolfe condition
    max_linesearch : int, optional (default=20)
        Maximum number of trials for the line search algorithm.

    SGD with L2 Parameters (l2sgd)
    ------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (1000 by default).
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization.
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion.
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this.
    calibration_eta : float, optional (default=0.1)
        Initial value of learning rate (eta) used for calibration.
    calibration_rate : float, optional (default=2.0)
        Rate of increase/decrease of learning rate for calibration.
    calibration_samples : int, optional (default=1000)
        Number of instances used for calibration.
    calibration_candidates : int, optional (default=10)
        Number of candidates of learning rate.
    calibration_max_trials : int, optional (default=20)
        Maximum number of trials of learning rates for calibration.

    Averaged Perceptron Parameters (ap)
    -----------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default).
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.

    Passive Aggressive Parameters (pa)
    ----------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default).
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.
    pa_type : int, optional (default=1)
        Strategy for updating feature weights:
            * 0: PA without slack variables
            * 1: PA type I
            * 2: PA type II
    c : float, optional (default=1)
        Aggressiveness parameter (used only for PA-I and PA-II).
    error_sensitive : bool, optional (default=True)
        Include square root of predicted incorrect labels into optimization routine.
    averaging : bool, optional (default=True)
        Compute average of feature weights at all updates.

    Adaptive Regularization of Weights Parameters (arow)
    ----------------------------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default).
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.
    variance : float, optional (default=1)
        Initial variance of every feature weight.
    gamma : float, optional (default=1)
        Trade-off between loss function and changes of feature weights.

    Returns
    -------
    Model
        A conditional random field trained on the dataset.
    """
    set_verbosity(verbose)

    if optimize_hyperparameters:
        if hyperparameters:
            LOGGER.warning(f"Specified hyperparameters will be overwritten: {hyperparameters}")

        # optionally tune hyperparameters first
        optimizer = HyperparameterOptimizer()
        results = optimizer.optimize_hyperparameters(dataset, labels, optimization_sample_size)

        # use hyperparameters of the best run
        hyperparameters = results[0]["hyperparameters"]

    # initialize trainer and start training
    trainer = Trainer(**hyperparameters)
    trainer.train(dataset, labels, model_filepath=str(model_filepath))

    # load and return the trained model
    return Model(model_filepath)
