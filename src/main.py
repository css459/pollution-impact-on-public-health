from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.etl.load import load_all
from src.etl.preprocess import split
from src.model.cluster import view_pca
from src.model.features import sma_featurizer


def fit_predict(x_tr, y_tr, x_te, y_te, label):
    gbm = GradientBoostingRegressor(n_estimators=200)
    gbm.fit(x_tr, y_tr)

    print('---', label, '-----------------------')

    for i in sorted(zip(gbm.feature_importances_, x_cols))[-5:]:
        print(i)

    print("MSE", mean_squared_error(y_te, gbm.predict(x_te)))
    print("R2 ", r2_score(y_te, gbm.predict(x_te)))

    # noise = np.random.normal(0, 1, len(x_test)) * 0.05
    #
    # print("With noise:")
    # print("MSE", mean_squared_error(y_test[:, 1], noise + gbm.predict(x_test)))
    # print("R2 ", r2_score(y_test[:, 1], noise + gbm.predict(x_test)))


def viz():
    """
    Visualize the initial clusters of age-adjusted cancer
    rates for each area. We will group each area
    by adding up all the years observed.

    :return:
    """
    grouped = df.groupby(by=['lat', 'lon']).sum()
    cancer_cols = [c for c in df.columns if "cancer" in c]
    print(cancer_cols)
    view_pca(grouped.drop(cancer_cols, 1), 'age-adjusted_rate',
             title="All Merged Data w. EMA, Colored by Age-Adjusted Cancer Rates\n\n",
             xlabel="First Component",
             ylabel="Second Component")


if __name__ == '__main__':
    # Load the merged data set
    df = load_all(from_file=False)

    # Featurize the merged data by add the Exponential
    # Moving Average for each column for 1, 2, and 3 year
    # windows as separate columns
    #
    # Reason: Assumption that places with consistently
    # high past past pollution will have higher cancer rates
    # that those with only high *current* pollution
    df = sma_featurizer(df, ema=True).dropna()


    # Scale, shuffle, split the merged data
    # We will consider the "cancer" columns
    # and "age-adjusted_rate" to be Y columns

    def _in_terms(t):
        y_terms = ['age-adjusted_rate', 'count', 'cancer', 'population']
        for ty in y_terms:
            if ty in t:
                return True
        return False


    y_cols = [c for c in df.columns if _in_terms(c)]
    x_cols = [c for c in df.columns if c not in y_cols]

    print("Considering X Columns:")
    print(x_cols)

    print("Considering Y Columns:")
    print(y_cols)

    # with open("features_fmt.txt", 'r') as fp:
    #     feats = [l.strip() for l in fp]
    # x_cols = feats

    x_train, y_train, x_test, y_test = split(df, y_cols=y_cols)

    for i in range(len(y_cols)):
        if 'ema' in y_cols[i] or 'sma' in y_cols[i]:
            continue

        fit_predict(x_train, y_train[:, i], x_test, y_test[:, i], y_cols[i])
