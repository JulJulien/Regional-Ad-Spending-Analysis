import numpy as np
import pandas as pd

np.random.seed(42)

regions = [
    "Northeast", "Southeast", "Midwest", "Southwest", "Northwest", "Europe",
    "Asia-Pacific", "South America", "Middle East", "Africa", "Central America", "Canada"
]

weeks = pd.date_range(start="2022-01-03", end="2024-12-30", freq="W-MON")

# --- Base structure ---
df = pd.DataFrame([(r, w) for r in regions for w in weeks], columns=["region", "week"])

# --- Generate correlated ad spend ---
base_spend = np.random.uniform(100, 300, len(df))

# add some correlation between spend channels
df["ad_spend_tv"] = base_spend + np.random.normal(0, 20, len(df))
df["ad_spend_digital"] = base_spend * np.random.uniform(0.7, 1.2, len(df))
df["ad_spend_outdoor"] = base_spend * np.random.uniform(0.3, 0.8, len(df))
df["ad_spend_retail"] = base_spend * np.random.uniform(0.4, 0.9, len(df))

# --- Environmental factors ---
region_temp_offset = {
    "Northeast": -5, "Southeast": +5, "Midwest": -10, "Southwest": +10, "Northwest": +0,
    "Europe": +0, "Asia-Pacific": +10, "South America": +12, "Middle East": +15,
    "Africa": +18, "Central America": +14, "Canada": -15
}

df["avg_temp"] = (
    60
    + np.random.normal(0, 10, len(df))
    + df["region"].map(region_temp_offset)
)

df["holiday"] = np.random.choice([0, 1], size=len(df), p=[0.92, 0.08])

# --- Competitor behavior (correlated with Coke spend) ---
df["pepsi_spend"] = df["ad_spend_tv"] * np.random.uniform(0.6, 1.4, len(df))

# --- Regional base performance ---
region_effect = {r: np.random.uniform(500, 1200) for r in regions}

# --- Sales simulation ---
def simulate_sales(row):
    base = region_effect[row["region"]]
    tv = np.log1p(row["ad_spend_tv"]) * 45
    digital = np.log1p(row["ad_spend_digital"]) * 60
    outdoor = np.log1p(row["ad_spend_outdoor"]) * 25
    retail = np.log1p(row["ad_spend_retail"]) * 30
    weather = 2.2 * (row["avg_temp"] - 60)
    holiday = 180 if row["holiday"] == 1 else 0
    competitor = -0.25 * row["pepsi_spend"]
    noise = np.random.normal(0, 100)
    return base + tv + digital + outdoor + retail + weather + holiday + competitor + noise

df["sales_volume"] = df.apply(simulate_sales, axis=1)
df["sales_volume"] = df["sales_volume"].clip(lower=200)

# --- Derived metrics ---
df["total_spend"] = (
    df["ad_spend_tv"] + df["ad_spend_digital"] +
    df["ad_spend_outdoor"] + df["ad_spend_retail"]
)
df["roi"] = df["sales_volume"] / df["total_spend"]

df = df.sort_values(["region", "week"]).reset_index(drop=True)

df.to_csv("cocacola_marketing_data.csv", index=False)

print(df.head())
print(f"\nDataset shape: {df.shape}")
