import random


def generate_random_colors(number_of_colors: int) -> list:
    """
    Generates random colors for each element in an interable of unknown length.
    Returns a dictionary where the keys are the elements in the list and the values are the random colors.

    Example:
    """
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF')
                            for j in range(6)]) for i in range(number_of_colors)]

    return colors



