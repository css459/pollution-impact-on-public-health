from src.etl.load import load_all
from src.etl.preprocess import split

# Scale, shuffle, split the merged data
df = load_all()
train, test = split(df)

# Visualize the initial clusters of age-adjusted cancer
# rates for each area. We will group each area
# by adding up all the years observed
from src.model.cluster import view_pca

grouped = df.groupby(by=['lat', 'lon']).sum()
cancer_cols = [c for c in df.columns if "cancer" in c]
print(cancer_cols)
view_pca(grouped.drop(['year'] + cancer_cols, 1), 'age-adjusted_rate',
         title="All Merged Data, Colored by Age-Adjusted Cancer Rates",
         xlabel="First Component",
         ylabel="Second Component")
