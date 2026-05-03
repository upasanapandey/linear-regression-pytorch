# generate_data.py
from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["target"] = housing.target  # median house price in $100k

df.to_csv("data.csv", index=False)
print(f"Saved {len(df)} rows, {df.shape[1]-1} features")
print(df.head())