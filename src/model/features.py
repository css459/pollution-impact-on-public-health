#
# Intro to ML Final Project
# Cole Smith
# features.py
#


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
    if 'lat' in by and 'lon' in by:
        df = df.drop('year', 1)

    g = df.groupby(by=by)

    # Define an inner convenience function
    def roll(l):
        if ema:
            r = g.sum().ewm(l).mean()
            r.columns = ["ema" + str(l) + "_" + c for c in r]
        else:
            r = g.expanding(l).sum()
            r.columns = ["sma" + str(l) + "_" + c for c in r]
        return r

    features = roll(lags[0])
    for lag in lags[1:]:
        tmp = roll(lag)
        features[tmp.columns] = tmp

    # Left join SMA/EMA features on original DataFrame
    features_indexes = features.index
    features = features.reset_index()
    for i in range(len(by)):
        b = by[i]
        features[b] = [ind[i] for ind in features_indexes]

    return df.merge(features, left_on=by, right_on=by)
