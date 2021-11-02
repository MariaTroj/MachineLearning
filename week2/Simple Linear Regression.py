from sklearn import linear_model
from week2_functions import *

if __name__ == "__main__":
    # read csv file
    df = pd.read_csv("..\\FuelConsumption.csv")

    # take a look at the dataset - display features names, ids and 5 first rows of data
    df.head()

    # summarize the data - count, mean, std, min, 25%, 50%, 75%, max for each feature
    df.describe()

    # plot histogram for each feature
    viz = df[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
    viz.hist()
    plt.show()

    # plot FUELCONSUMPTION_COMB and ENGINESIZE against CO2EMISSIONS
    plt.scatter(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS, color='blue')
    plt.xlabel("FUELCONSUMPTION_COMB")
    plt.ylabel("Emission")
    plt.show()

    plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    train_x, test_x, train_y, test_y = train_test_split(df[['ENGINESIZE']], df[['CO2EMISSIONS']],
                                                        test_size=0.2)

    # model the data
    # calculate coefficients for simple linear regression (ax + b), x-axis is engine size,
    # y-axis is co2 emission.
    regr = linear_model.LinearRegression()

    regr.fit(train_x, train_y)
    # The coefficients
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)

    # plot the fit line and the data
    plt.scatter(train_x.ENGINESIZE, train_y.CO2EMISSIONS, color='blue')
    plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    metrics(test_y, regr.predict(test_x))