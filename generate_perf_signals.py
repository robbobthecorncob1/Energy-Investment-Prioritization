import pandas as pd

def calc_base_load(series: pd.Series) -> float:
    """
    Calculates the base load for a given data series.

    Args:
        series: A Panda Series containing numeric values.

    Returns:
        A float representing the 10th percentile of the series.
    """

    return series.quantile(0.10)

def calc_sensitivity(group: pd.DataFrame) -> float:
    """
    Calculates the Pearson correlation coefficiant.

    Args:
        group: A Panda Data Frame

    Returns:
        A float representing the Pearson correlation coefficiant.
    """
    
    if len(group) < 24: 
        return 0
    return group["readingvalue"].corr(group["temperature_2m"])

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds and removes outliers from the data frame.
    
    Args:
        df: A Pandas Data Frame.

    Returns:
        A modified Panda Data Frame with the outliers removed.
    """
    Q1 = df['mean_eui'].quantile(0.25)
    Q3 = df['mean_eui'].quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + (1.5 * IQR)
    lower_fence = Q1 - (1.5 * IQR)

    return df[(df['mean_eui'] <= upper_fence) & (df['mean_eui'] >= lower_fence)].copy()


print("Loading cleaned hourly data...", end="")
cleanData_df = pd.read_csv("processed/hourly_electricity_cleaned.csv")
print("Done!")

# Create a data frame object containing the performance signals
signals = cleanData_df.groupby("simscode").agg(
    grossarea=("grossarea", "first"),
    total_kwh=("readingvalue", "sum"),
    mean_eui=("energyuseintensity", "mean"),
    base_load_kwh=("readingvalue", calc_base_load)
).reset_index()

print("Analyzing weather sensitivity...", end="")
sensitivity = cleanData_df.groupby("simscode").apply(calc_sensitivity, include_groups=False).reset_index()
sensitivity.columns = ["simscode", "weather_sensitivity"]
print("Done!")

scores_df = pd.merge(signals, sensitivity, on="simscode")
scores_df["avg_hourly_kwh"] = scores_df["total_kwh"] / cleanData_df["hour_timestamp"].nunique()
scores_df["base_load_ratio"] = scores_df["base_load_kwh"] / scores_df["avg_hourly_kwh"]

scores_df = remove_outliers(scores_df)
scores_df = scores_df.sort_values(by="mean_eui", ascending=False)

scores_df.to_csv("processed/building_perf_signals.csv", index=False)
print(scores_df[["simscode", "mean_eui", "weather_sensitivity", "base_load_ratio"]].head())