#
# Intro to ML Final Project
# Cole Smith
# features.py
#

from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNetCV


def sma_featurizer(df, by=None, lags=None, ema=True):
    """
    Computes the SMA of each column, grouped by
    the `by` set. This follows for each `i` in
    `lags`. The default `by` is `[lat, lon]`. The
    default `lags` is `[1,2,3]`, which represent years
    in the merged dataset in this project. SMA
    columns will be appended with prefix smaN_.

    :param df:      DataFrame to featurize
    :param by:      Grouping columns (default: [lat, lon]
    :param lags:    SMA i periods (default: [1,2,3])
    :param ema:     Use exponential moving averages
    :return:        Featurized DataFrame
    """
    # Resolve defaults
    if by is None:
        by = ['lat', 'lon']
    if lags is None:
        lags = [1, 2, 3]

    # Re-index DataFrame to by columns plus year
    df = df.set_index(by + ['year'])

    # Group by without year to compute SMA/EMA
    g = df.groupby(by=by)

    # Define an inner convenience function
    def roll(i):

        # Apply the SMA/EMA to the inner groupings, which should
        # contain "year"
        if ema:
            r = g.apply(lambda x: x.sort_values(by='year').ewm(i).mean())
            r.columns = ["ema" + str(i) + "_" + c for c in r.columns]
        else:
            r = g.apply(lambda x: x.sort_values(by='year').expanding(i).mean())
            r.columns = ["sma" + str(i) + "_" + c for c in r.columns]
        return r

    # Compute for all lags
    features = roll(lags[0])
    for lag in lags[1:]:
        tmp = roll(lag)

        # Join on by plus year, since year is also still in index
        features = features.join(tmp, how='inner')

    # Join on the original DataFrame, sort, return
    return df.join(features, how='inner').sort_index()


def elastic_feature_select(x_train, y_train, elastic_alpha=1.0, elastic_l1_ratio=0.5):
    """
    Perform a Cross-Validated Recursive Feature Elimination
    procedure using an ElasticNet regressor. NOTE:
    only deliver ONE feature for y_train. This function
    does not implement a One-vs-All strategy.

    :param x_train:             Design matrix
    :param y_train:             Y values
    :param elastic_alpha:       Alpha for ElasticNet model
    :param elastic_l1_ratio:    L1 Ratio for ElasticNet model
    :return:                    Array of indexes of the best found features
    """
    enet = ElasticNetCV(n_jobs=-1, cv=5)
    rfe = RFECV(enet, scoring='r2', n_jobs=-1, verbose=2)
    rfe.fit(x_train, y_train)
    return rfe
