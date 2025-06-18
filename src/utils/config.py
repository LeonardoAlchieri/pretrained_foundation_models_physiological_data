from omegaconf import DictConfig

def check_aggregator(aggregator: object | str | DictConfig) -> object | str:
    """
    Checks if the aggregator is a valid callable or a string.

    Parameters
    ----------
    aggregator : object | str
        The aggregator to check.

    Returns
    -------
    object
        The aggregator if it is callable, otherwise raises an error.
    """
    if isinstance(aggregator, DictConfig):
        if "_value_" in aggregator.keys():
            return aggregator['_value_']
        else:
            raise ValueError(f"Invalid aggregator configuration. Received {aggregator}.")
    else:
        return aggregator
        