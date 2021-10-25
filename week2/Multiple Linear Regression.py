from sklearn import linear_model
from week2_functions import *


if __name__ == "__main__":
    df = pd.read_csv("..\\FuelConsumption.csv")

    train_x, test_x, train_y, test_y = train_test_split(df[['ENGINESIZE', 'CYLINDERS',
                                                             'FUELCONSUMPTION_COMB']],
                                                        df[['CO2EMISSIONS']], 0.8)

    # model the data
    # calculate coefficients for simple linear regression (a0 + a1x1 + a2x2 +...+ anxn)
    regr = linear_model.LinearRegression()

    regr.fit(train_x, train_y)
    # The coefficients
    print('Coefficients: ', regr.coef_)

    metrics(test_y, regr.predict(test_x))
    # variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(test_x, test_y))