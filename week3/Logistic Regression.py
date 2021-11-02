import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, log_loss
from week3.week3_functions import *

def X_y_split_normalize(dataframe, x_labels, y_label):
    X = dataframe[x_labels].values
    y = dataframe[y_label].values
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, y


if __name__ == "__main__":
    churn_df = pd.read_csv("..\\ChurnData.csv")
    churn_df['churn'] = churn_df['churn'].astype('int')

    X, y = X_y_split_normalize(churn_df, ['tenure', 'age', 'address', 'income', 'ed', 'employ',
                                          'equip'], 'churn')

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

    # Regularization is a technique used to solve the overfitting problem of ML models.
    # C parameter indicates inverse of regularization strength which must be a positive float.
    # Smaller values specify stronger regularization
    LR = LogisticRegression(C=0.01, solver='liblinear')
    LR.fit(X_train, y_train)
    predict_y = LR.predict(X_test)
    predict_y_prob = LR.predict_proba(X_test)
    print("Jaccard score = ", jaccard_score(y_test, predict_y, pos_label=0))

    cnf_matrix = confusion_matrix(y_test, predict_y, labels=[1, 0])
    plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False,
                          title='Confusion matrix')
    #Precision is a measure of the accuracy provided that a class label has been predicted.
    # It is defined by: precision = TP / (TP + FP)
    # Recall is the true positive rate. It is defined as: Recall =  TP / (TP + FN)
    print(classification_report(y_test, predict_y))
    print('Log loss: ', log_loss(y_test, predict_y_prob))


