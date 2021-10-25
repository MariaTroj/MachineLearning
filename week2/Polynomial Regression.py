from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from week2_functions import *

if __name__ == "__main__":
    df = pd.read_csv("..\\FuelConsumption.csv")

    train_x, test_x, train_y, test_y = train_test_split(df[['ENGINESIZE']], df[['CO2EMISSIONS']],
                                                        0.8)

    """
    PolynomialFeatures() drives a new feature sets from the original feature set. For example, the 
    original feature set has only one feature, x1. If we select the degree of the polynomial
    to be 2, then fit_transform function will generate matrix with all 3 features:
     - degree=0: x1^0 = 1, 
     - degree=1: x1^1 = x1,
     - degree=2: x1^2
    """
    poly = PolynomialFeatures(degree=2)
    train_x_poly = poly.fit_transform(train_x)
    test_x_poly = poly.fit_transform(test_x)

    # now, problem may be presented as multiple linear regression problem: y = A^T *
    # train_x_poly, where A^T = [a0, a1, a2],
    # y = a0 * train_x_poly[0] + a1 * train_x_poly[1] + a2 * train_x_poly[2]
    # Therefore, this polynomial regression is considered to be a special case of traditional
    # multiple linear regression. So, the same mechanism as linear regression can be used to
    # solve such a problems.
    clf = linear_model.LinearRegression()
    train_y_ = clf.fit(train_x_poly, train_y)
    predict_y = clf.predict(test_x_poly)
    print('Coefficients: ', clf.coef_)
    print('Intercept: ', clf.intercept_)

    plt.scatter(train_x.ENGINESIZE, train_y.CO2EMISSIONS, color='blue')
    XX = np.arange(0.0, 10.0, 0.1) # XX is of type numpy.ndarray, shape (100, 1) and is filled
    # with floats (0.0, 0.1, 0.2, ... 9.9)
    yy = clf.intercept_[0] + clf.coef_[0][1] * XX + clf.coef_[0][2] * np.power(XX, 2)
    plt.plot(XX, yy, '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    metrics(test_y, predict_y)