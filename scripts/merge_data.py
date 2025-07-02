import pandas as pd

# Load satellite and ground files
sat = pd.read_csv("data/satellite/all_pollutants_sat.csv")
ground = pd.read_csv("data/ground/all_pollutants_ground.csv")

# Merge on common columns
merged = pd.merge(sat, ground, on=["date", "city", "country"], how="inner")

# Save to final merged file
merged.to_csv("data/merged_pollution_data.csv", index=False)
print("✅ Merged file saved to data/merged_pollution_data.csv")
