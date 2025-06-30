from numpy import ndarray, stack, nanmean, nanstd, array
from pandas import Series, DataFrame
from cvxEDA import cvxEDA
from gc import collect as pick_up_trash

from logging import getLogger


logger = getLogger("eda")

# See https://github.com/lciti/cvxEDA for more EDA analysis methdos

# TODO: probably remove and use some third party library
def standardize(signal: Series | ndarray | list) -> ndarray:
    """Simple method to standardize an EDA signal.

    Parameters
    ----------
    signal : Series | ndarray | list
        signal to standardize

    Returns
    -------
    ndarray
        returns an array standardized
    """
    y: ndarray = array((signal))

    yn: ndarray = (y - nanmean(y)) / nanstd(y)
    return yn


def decomposition(
    eda_signal: ndarray, frequency: int = 4, **kwargs
) -> dict[str, ndarray]:
    """This method will apply the cvxEDA decomposition to an EDA signal. The cvxEDA
    implementation is the one from Greco et al.

    Parameters
    ----------
    eda_signal : Series | ndarray | list
        eda signal to be decomposed
    frequency : int, optional
        frequency of the input signal, e.g. 64Hz

    Returns
    -------
    dict[str, ndarray]
        the method returns a dictionary with the decomposed signals
        (see cvxEDA for more details)
    """
    
    yn = standardize(signal=eda_signal)
    return cvxEDA(yn, 1.0 / frequency)
