import pandas as pd

df = pd.read_csv("data/merged_pollution_data.csv")
print("\n📋 Columns in your merged_pollution_data.csv:")
print(df.columns.tolist())
