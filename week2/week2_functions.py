import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# different model evaluation metrics:
    #  - Mean Absolute Error: mean of the absolute value of the errors. The easiest to understand
    #  - Mean Squared Error (MSE): mean of the squared error. The squared term exponentially
    #    increases larger errors in comparison to smaller ones.
    #  - Root Mean Squared Error (RMSE).
    #  - R-squared - is not an error, but rather a popular metric to measure the performance of
    #    regression model. It represents how close the data points are to the fitted regression
    #    line. The higher the R-squared value, the better the model fits the data. The best possible
    #    score is 1.0 and it can be negative (because the model can be arbitrarily worse).
def metrics(test_y, predict_y):
    print("Mean absolute error (MAE): %.2f" % np.mean(np.absolute(predict_y - test_y)))
    print("Residual sum of squares (MSE): %.4f" % np.mean((predict_y - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y, predict_y))



# split dataset into train and test sets: 80% of the entire dataset will be used for
    # training and 20% for testing.
    # np.random.rand() returns floats from range (0, 1)
    # np.random.rand(a) returns ax1 ndarray filled with floats from range (0, 1)
    # np.random.rand(a) < b ax1 ndarray filled with True if random float is less than b and False
    # otherwise
def custom_train_test_split(x_data, y_data, trigger):
    msk = np.random.rand(len(x_data)) < trigger
    train_x = x_data[msk]
    test_x = x_data[~msk]
    train_y = y_data[msk]
    test_y = y_data[~msk]

    return train_x, test_x, train_y, test_y