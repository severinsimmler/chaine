"""
chaine.validation
~~~~~~~~~~~~~~~~~

This module implements functions to validate input sequences (either for training or inference).
"""

from chaine.typing import Sequence

# supported feature value data types
TYPES = (str, int, float, bool)


def is_valid_sequence(sequence: Sequence) -> bool:
    """Check if the given sequence has valid input format.

    Parameters
    ----------
    sequence : Sequence
        Sequence to validate.

    Returns
    -------
    bool
        True if sequence is valid, False otherwise.
    """
    return isinstance(sequence, list) and all(is_valid_token(token) for token in sequence)


def is_valid_token(token: dict) -> bool:
    """Check if the given token has valid input format.

    Parameters
    ----------
    token : dict
        Token to validate.

    Returns
    -------
    bool
        True if sequence is valid, False otherwise.
    """
    return isinstance(token, dict) and all(isinstance(value, TYPES) for value in token.values())
