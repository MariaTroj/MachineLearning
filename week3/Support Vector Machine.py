import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from week3.week3_functions import *
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score


def X_y_split_normalize(dataframe, x_labels, y_label):
    X = dataframe[x_labels].values
    y = dataframe[y_label].values
    return X, y


if __name__ == "__main__":
    cell_df = pd.read_csv("..\\cell_samples.csv")

    ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize',
                                                   color='DarkBlue', label='malignant')
    cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize',
                                              color='Yellow', label='benign', ax=ax)
    plt.show()

    # some values in BareNuc ar enot numerical. Drop these rows
    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

    X, y = X_y_split_normalize(cell_df,
                               ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
                                'BareNuc', 'BlandChrom', 'NormNucl', 'Mit'], 'Class')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    predict_y = clf.predict(X_test)

    cnf_matrix = confusion_matrix(y_test, predict_y, labels=[2, 4])
    np.set_printoptions(precision=2)

    print(classification_report(y_test, predict_y))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=False,
                          title='Confusion matrix')

    print("F1 score: ", f1_score(y_test, predict_y, average='weighted'))
    print("Jaccard score: ", jaccard_score(y_test, predict_y, pos_label=2))