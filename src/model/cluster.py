#
# Intro to ML Final Project
# Cole Smith
# cluster.py
#

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def view_pca(df, emphasize_column=None, title=None, xlabel=None, ylabel=None):
    """
    Charts the 2-component PCA decomposition of
    the given DataFrame, with optional
    coloring upon a separate column. This
    column will NOT be part of the PCA decomposition.

    :param df:                  DataFrame to decompose
    :param emphasize_column:    Column for coloring PCA
    :param title:               Title of Plot
    :param xlabel:              X axis label
    :param ylabel:              Y axis label
    :return:                    `None`
    """
    c = None
    if emphasize_column:
        c = df[emphasize_column]
        df = df.drop(emphasize_column, 1)

    pca = PCA().fit_transform(df)
    plt.scatter(pca[:, 0], pca[:, 1], c=c)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
