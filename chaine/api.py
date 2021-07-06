"""
chaine.api
~~~~~~~~~~

This module implements the high-level API to train a conditional random field
"""

from chaine.crf import Model, Trainer
from chaine.typing import Dataset, Filepath, Labels


def train(
    dataset: Dataset,
    labels: Labels,
    model_filepath: Filepath = "model.crf",
    **kwargs,
) -> Model:
    """Train a conditional random field

    Parameters
    ----------
    dataset : Dataset
        Dataset consisting of sequences of feature sets
    labels : Labels
        Labels corresponding to each instance in the dataset
    model_filepath : Filepath
        Path to model location
    algorithm : str
        Following algorithms are available:
            * lbfgs: Limited-memory BFGS with L1/L2 regularization
            * l2sgd: Stochastic gradient descent with L2 regularization
            * ap: Averaged perceptron
            * pa: Passive aggressive
            * arow: Adaptive regularization of weights

    Limited-memory BFGS Parameters
    ------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (unlimited by default)
    num_memories : int, optional (default=6)
        Number of limited memories for approximating the inverse hessian matrix
    c1 : float, optional (default=0)
        Coefficient for L1 regularization
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this
    linesearch : str, optional (default="MoreThuente")
        Line search algorithm used in updates:
            * MoreThuente: More and Thuente's method
            * Backtracking: Backtracking method with regular Wolfe condition
            * StrongBacktracking: Backtracking method with strong Wolfe condition
    max_linesearch : int, optional (default=20)
        Maximum number of trials for the line search algorithm

    SGD with L2 Parameters
    ----------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (1000 by default)
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this
    calibration_eta : float, optional (default=0.1)
        Initial value of learning rate (eta) used for calibration
    calibration_rate : float, optional (default=2.0)
        Rate of increase/decrease of learning rate for calibration
    calibration_samples : int, optional (default=1000)
        Number of instances used for calibration
    calibration_candidates : int, optional (default=10)
        Number of candidates of learning rate
    calibration_max_trials : int, optional (default=20)
        Maximum number of trials of learning rates for calibration

    Averaged Perceptron Parameters
    ------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default)
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence

    Passive Aggressive Parameters
    -----------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default)
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence
    pa_type : int, optional (default=1)
        Strategy for updating feature weights:
            * 0: PA without slack variables
            * 1: PA type I
            * 2: PA type II
    c : float, optional (default=1)
        Aggressiveness parameter (used only for PA-I and PA-II)
    error_sensitive : bool, optional (default=True)
        Include square root of predicted incorrect labels into optimization routine
    averaging : bool, optional (default=True)
        Compute average of feature weights at all updates

    Adaptive Regularization of Weights (AROW) Parameters
    ----------------------------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default)
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence
    variance : float, optional (default=1)
        Initial variance of every feature weight
    gamma : float, optional (default=1)
        Trade-off between loss function and changes of feature weights

    Returns
    -------
    Model
        A conditional random field trained on the dataset
    """
    # initialize trainer and start training
    trainer = Trainer(**kwargs)
    trainer.train(dataset, labels, model_filepath=str(model_filepath))

    # load and return the trained model
    return Model(model_filepath)
