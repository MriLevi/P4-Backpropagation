import math

def sigmoid(input):
    """Applies sigmoid function to the input"""
    return 1/(1+math.exp(-input))

def binary_step(input):
    """Returns 0 if the input is less than or equal to zero otherwise returns 1 """
    if input <= 0:
        return 0
    else:
        return 1

def linear(input):
    """
    y = f(x)
    Linear activation returns the input as is
    """
    return input
