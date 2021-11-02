import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, jaccard_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from week3.week3_functions import *


def prepare_data(filename):
    df = pd.read_csv(filename)

    # convert date to date time object
    df['due_date'] = pd.to_datetime(df['due_date'])
    df['effective_date'] = pd.to_datetime(df['effective_date'])

    # 260 people have paid off the loan on time while 86 have gone into collection
    # df['loan_status'].value_counts()
    '''
    # plot principal and loan status due to gender
    bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
    g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
    g.map(plt.hist, 'Principal', bins=bins, ec="k")
    g.axes[-1].legend()
    plt.show()

    # plot age and loan status due to gender
    bins = np.linspace(df.age.min(), df.age.max(), 10)
    g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
    g.map(plt.hist, 'age', bins=bins, ec="k")
    g.axes[-1].legend()
    plt.show()
    '''
    df['dayofweek'] = df['effective_date'].dt.dayofweek
    # bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
    # g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
    # g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
    # g.axes[-1].legend()
    # plt.show() # people how get the loan at the end of the week don't pay it off

    # use Feature binarization to set a threshold value less than day 4
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)

    # 86 % of female pay there loans while only 73 % of males pay there loan
    # df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

    df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)

    Feature = df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
    Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
    Feature.drop(['Master or Above'], axis=1, inplace=True)

    X = Feature
    X = preprocessing.StandardScaler().fit(X).transform(X)
    y = df['loan_status'].values

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

def k_neighbors_find_best_k(k_max, X_train, X_test, y_train, y_test):
    acc_score = np.zeros((k_max - 1))
    for k in range(1, k_max):
        acc_score[k - 1] = k_neighbors(k, X_train, X_test, y_train, y_test)

    plt.plot(range(1, k_max), acc_score, 'g')
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.show()
    return np.argmax(acc_score)

def decision_tree(X_train, X_test, y_train, y_test):
    loan_tree = DecisionTreeClassifier(criterion="entropy")
    loan_tree.fit(X_train, y_train)

    y_pred = loan_tree.predict(X_test)

    return metrics.accuracy_score(y_test, y_pred)

def support_vector_machine(X_train, X_test, y_train, y_test):
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred, labels=['COLLECTION', 'PAIDOFF'])
    print(classification_report(y_test, y_pred, labels=['COLLECTION', 'PAIDOFF']))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['COLLECTION', 'PAIDOFF'], normalize=False,
                          title='Confusion matrix')
    return f1_score(y_test, y_pred, average='weighted'), jaccard_score(y_test, y_pred, pos_label='COLLECTION')

def logistic_regr(X_train, X_test, y_train, y_test):
    log_regr = LogisticRegression(C=0.01, solver='liblinear')
    log_regr.fit(X_train, y_train)
    predict_y = log_regr.predict(X_test)
    predict_y_prob = log_regr.predict_proba(X_test)


    cnf_matrix = confusion_matrix(y_test, predict_y, labels=['COLLECTION', 'PAIDOFF'])
    plot_confusion_matrix(cnf_matrix, classes=['COLLECTION', 'PAIDOFF'], normalize=False,
                          title='Confusion matrix')
    # Precision is a measure of the accuracy provided that a class label has been predicted.
    # It is defined by: precision = TP / (TP + FP)
    # Recall is the true positive rate. It is defined as: Recall =  TP / (TP + FN)
    print(classification_report(y_test, predict_y))
    return log_loss(y_test, predict_y_prob), jaccard_score(y_test, predict_y, pos_label='COLLECTION')

if __name__ == "__main__":
    X, y = prepare_data('..\\loan_train.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

    # print("Best accuracy is for k =", k_neighbors_find_best_k(15, X_train, X_test, y_train,
    #                                                           y_test) + 1)

    # print(f"DecisionTrees's Accuracy:", decision_tree(X_train, X_test, y_train, y_test))

    # f1_s, jaccard_s = support_vector_machine(X_train, X_test, y_train, y_test)
    # print("F1 score: ", f1_s)
    # print("Jaccard score: ", jaccard_s)

    log_loss, jaccard_s = logistic_regr(X_train, X_test, y_train, y_test)
    print('Log loss: ', log_loss)
    print("Jaccard score: ", jaccard_s)
