import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# evalution metrics
def MSE():
    # TODO: Sarthak
    pass

def RMSE(predicted_values, expected_values):
    """
    Root Mean Square Error
    rmse = sqrt(1/N * sum(1, N)((y_n - t_n)^2)
    input parameters:
        predicted_values : Predicted Values
        expected_values   : Expected Values

    output:
        rmse(float) : Root mean square error for the given input values
    """
    # TODO: expected error checks and type checking
    N = len(predicted_values)
    sum = 0
    for n in range(N):
        sum += (predicted_values[n] - expected_values[n])
    return math.sqrt(1/N * sum)
    # TODO: Verify and test the equation and implementation

def R2():
    # TODO: Abhinav
    pass


