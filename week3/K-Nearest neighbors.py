import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def X_y_split_normalize(dataframe, x_labels, y_label):
    X = dataframe[x_labels].values
    y = dataframe[y_label].values
    # normalize the data
    # Data Standardization gives the data zero mean and unit variance, it is good practice,
    # especially for algorithms such as KNN which is based on the distance of data points:
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, y


def k_neighbors(k, X_train, X_test, y_train, y_test, verbose = False):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_predict = neigh.predict(X_test)

    # metrics.accuracy_score is the same that Jaccard index.
    # the closer the index is to 1, the higher the accuracy
    if verbose:
        print(f"Train set Accuracy for k = {k}: ", metrics.accuracy_score(y_train, neigh.predict(
            X_train)))
        print(f"Test set Accuracy for k = {k}: ", metrics.accuracy_score(y_test, y_predict))

    return metrics.accuracy_score(y_test, y_predict)

if __name__ == "__main__":
    df = pd.read_csv('..\\teleCust1000t.csv')

    print(df['custcat'].value_counts())

    df.hist(column='income', bins=50)
    plt.show()

    X, y = X_y_split_normalize(df, ['region', 'tenure', 'age', 'marital', 'address', 'income',
                                    'ed', 'employ', 'retire', 'gender', 'reside'], 'custcat')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    k_max = 10
    mean_acc = np.zeros((k_max - 1))
    std_acc = np.zeros((k_max - 1))
    for k in range(1, k_max):
        mean_acc[k - 1] = k_neighbors(k, X_train, X_test, y_train, y_test)

    plt.plot(range(1, k_max), mean_acc, 'g')
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.show()