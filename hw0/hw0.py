import numpy as np


def draw_samples(n: int):
    """
    Generate n random samples from set X where P (X = 1) = 0.35, P (X = 0) = 0.45
and P (X = −1) = 0.2
    :param n: integer
        number of samples generated
    :return: 1d array with shape n
        n randomly generated samples
    """
    x = [-1, 0, 1]
    probs = [0.2, 0.45, 0.35]

    result = np.random.choice(a=x, size=n, p=probs)

    return result


def sum_squares(arr):
    """
    Return the sum of squares of the given 1d array
    :param arr: 1d array
        given array
    :return: int
        sum of squares of arr
    """
    result = np.dot(a=arr, b=arr)
    return result


def troublemakers(n: int):
    """
    Return the liquid count in apple juice cup and oreo milkshake cup respectively after n rounds
    :param n: int
        number of rounds
    :return: 1d array
        final liquid count in apple juice cup and oreo milkshake cup respectively
    """
    result = np.array([1.0, 1.0])

    for i in range(n):
        apple = result[0] * 0.35
        np.put(result, 0, result[0] - apple)
        np.put(result, 1, result[1] + apple)
        oreo = result[1] * 0.2
        np.put(result, 1, result[1] - oreo)
        np.put(result, 0, result[0] + oreo)

    return result


# for debugging
# if __name__ == "__main__":
#     # a = draw_samples(10)
#     # a = sum_squares([1, 2, 3, 4])
#     a = troublemakers(10)
#     print(a)