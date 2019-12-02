#
# Intro to ML Final Project
# Cole Smith
# features.py
#

from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNet


def sma_featurizer(df, by=None, lags=None, ema=True):
    """
    Computes the SMA of each column, grouped by
    the `by` set. This follows for each `lag` in
    `lags`. The default `by` is `[lat, lon]`. The
    default `lags` is `[1,2,3]`, which represent years
    in the merged dataset in this project. SMA
    columns will be appended with prefix smaN_.

    :param df:      DataFrame to featurize
    :param by:      Grouping columns (default: [lat, lon]
    :param lags:    SMA lag periods (default: [1,2,3])
    :param ema:     Use exponential moving averages
    :return:        Featurized DataFrame
    """
    # Resolve defaults
    if by is None:
        by = ['lat', 'lon']
    if lags is None:
        lags = [1, 2, 3]

    # Drop year if we're grouping by lat and lon
    # if 'lat' in by and 'lon' in by:
    #     df = df.drop('year', 1)

    g = df.groupby(by=by)

    # TODO Fix EWM on groupby

    # Define an inner convenience function
    def roll(l):
        if ema:
            r = g.apply(lambda x: x.ewm(l).mean())
            r.columns = by + ["ema" + str(l) + "_" + c for c in r.columns if c not in by]
        else:
            r = g.expanding(l).sum()
            r.columns = by + ["sma" + str(l) + "_" + c for c in r.columns if c not in by]
        return r

    # Compute for all lags
    features = roll(lags[0])
    for lag in lags[1:]:
        tmp = roll(lag)
        features = features.merge(tmp, on=by, how='inner')

    # Left join SMA/EMA features on original DataFrame
    features_indexes = features.index
    features = features.reset_index()
    for i in range(len(by)):
        b = by[i]
        features[b] = [ind[i] for ind in features_indexes]

    return df.merge(features, on=by, how='inner')


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
    enet = ElasticNet(alpha=elastic_alpha, l1_ratio=elastic_l1_ratio)

    rfe = RFECV()
