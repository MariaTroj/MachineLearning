from scipy.optimize import curve_fit
from week2_functions import *

# Sigmoid/Logistic function
# startS with a slow growth, increase growth in the middle, and decrease at the end
# 𝛽_1 : Controls the curve's steepness,
# 𝛽_2 : Slides the curve on the x-axis.
def sigmoid(x, beta_1, beta_2):
    y = 1 / (1 + np.exp(-beta_1 * (x - beta_2)))
    return y


if __name__ == "__main__":
    df = pd.read_csv("..\\china_gdp.csv")

    plt.figure(figsize=(8, 5))
    x_data, y_data = (df["Year"].values, df["Value"].values)
    plt.plot(x_data, y_data, 'ro')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()

    X = np.arange(-5.0, 5.0, 0.1)
    Y = sigmoid(X, 1, 2)
    plt.plot(X, Y)
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.show()

    # normalize the data
    xdata = x_data / max(x_data)
    ydata = y_data / max(y_data)

    # curve_fit uses non-linear least squares to fit our sigmoid function, to the data.
    # The sum of the squared residuals of sigmoid(xdata, *popt) - ydata is minimized.
    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    # print the final parameters
    print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

    x = np.linspace(1960, 2015, 55)
    x = x / max(x)
    plt.figure(figsize=(8, 5))
    y = sigmoid(x, *popt)
    plt.plot(xdata, ydata, 'ro', label='data')
    plt.plot(x, y, linewidth=3.0, label='fit')
    plt.legend(loc='best')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()

    train_x, test_x, train_y, test_y = train_test_split(xdata, ydata, test_size=0.2)
    popt, pcov = curve_fit(sigmoid, train_x, train_y)
    y_predict = sigmoid(test_x, *popt)
    metrics(test_y, y_predict)
