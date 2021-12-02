from chaine.typing import Filepath


def parse_transitions(filepath: Filepath) -> dict[str, float]:
    """Parse transition weights from an exported model.

    Parameters
    ----------
    filepath : Filepath
        Path to the exported model.

    Returns
    -------
    dict[str, float]
        Mapping from transition to weight.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        components = f.read().split("\n\n")

        # transitions are in the fourth dictionary
        transitions = components[3].split("\n")[1:-1]

        return {
            transition.split(":")[0].removeprefix("  (1) "): float(transition.split(":")[1])
            for transition in transitions
        }


def parse_state_features(filepath: Filepath) -> dict[str, float]:
    """Parse state feature weights from an exported model.

    Parameters
    ----------
    filepath : Filepath
        Path to the exported model.

    Returns
    -------
    dict[str, float]
        Mapping from state feature to weight.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        components = f.read().split("\n\n")

        # state features are in the fifth dictionary
        state_features = components[4].split("\n")[1:-1]

        return {
            ":".join(state.split(":")[:-1]).removeprefix("  (0) "): float(state.split(":")[-1])
            for state in state_features
        }
