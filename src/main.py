from src.etl.load import load_all
from src.etl.preprocess import split
from src.model.features import sma_featurizer

# Load the merged data set
df = load_all(from_file=True)

# Featurize the merged data by add the Exponential
# Moving Average for each column for 1, 2, and 3 year
# windows as separate columns
#
# Reason: Assumption that places with consistently
# high past past pollution will have higher cancer rates
# that those with only high *current* pollution
df = sma_featurizer(df, ema=False)

# Scale, shuffle, split the merged data
# We will consider the "cancer" columns
# and "age-adjusted_rate" to be Y columns
y_cols = ['age-adjusted_rate'] + [c for c in df.columns if "cancer" in c]
x_cols = [c for c in df.columns if c not in y_cols]

print("Considering X Columns:")
print(x_cols)

print("Considering Y Columns:")
print(y_cols)

x_train, y_train, x_test, y_test = split(df, y_cols=y_cols)

# Visualize the initial clusters of age-adjusted cancer
# rates for each area. We will group each area
# by adding up all the years observed
from src.model.cluster import view_pca

grouped = df.groupby(by=['lat', 'lon']).sum()
cancer_cols = [c for c in df.columns if "cancer" in c]
print(cancer_cols)
view_pca(grouped.drop(cancer_cols, 1), 'age-adjusted_rate',
         title="All Merged Data w. EMA, Colored by Age-Adjusted Cancer Rates\n\n",
         xlabel="First Component",
         ylabel="Second Component")
