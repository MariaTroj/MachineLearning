import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def X_y_split(dataframe, x_labels, y_label):
    X = dataframe[x_labels].values
    y = dataframe[y_label].values

    return X, y


if __name__ == "__main__":
    my_data = pd.read_csv("..\\drug200.csv", delimiter=",")
    X, y = X_y_split(my_data, ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], "Drug")

    # Some features in this dataset are categorical, such as Sex or BP. Sklearn Decision Trees does
    # not handle categorical variables.
    le = LabelEncoder()
    X[:, 1] = le.fit_transform(X[:, 1])
    X[:, 2] = le.fit_transform(X[:, 2])
    X[:, 3] = le.fit_transform(X[:, 3])


    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3,
                                                                    random_state=3)

    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    drugTree.fit(X_trainset, y_trainset)

    predTree = drugTree.predict(X_testset)

    print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
