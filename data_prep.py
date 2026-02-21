import pandas as pd

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds and removes outliers from the data frame.
    
    Args:
        df: A Pandas Data Frame.

    Returns:
        A modified Panda Data Frame with the outliers removed.
    """
    Q1 = df['energyuseintensity'].quantile(0.25)
    Q3 = df['energyuseintensity'].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = max(0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR

    return df[(df['energyuseintensity'] >= lower_bound) & (df['energyuseintensity'] <= upper_bound)]

def run_data_prep():
    """
    Executes the data ingestion, cleaning, and feature engineering pipeline.
    """

    # Read in building, weather, and meter data
    print(f"Reading in data...", end="")
    building_df = pd.read_csv("data/building_metadata.csv")
    weather_df = pd.read_csv("data/weather-sept-oct-2025.csv", parse_dates=["date"])
    meter_sept = pd.read_csv("data/meter-data-sept-2025.csv", parse_dates=["readingtime"])
    meter_oct = pd.read_csv("data/meter-data-oct-2025.csv", parse_dates=["readingtime"])
    print(f"Done!")

    print("Cleaning up data...", end="")
    meter_df = pd.concat([meter_sept, meter_oct], ignore_index=True)
    del meter_sept
    del meter_oct

    meterElec_df = meter_df[meter_df["utility"] == "ELECTRICITY"].copy()
    del meter_df
    meterElec_df["hourly_est_kwh"] = meterElec_df["readingvalue"] * 4
    meterElec_df["hour_timestamp"] = meterElec_df["readingtime"].dt.floor("h")

    weather_df["hour_timestamp"] = weather_df["date"].dt.floor("h")
    weatherSubset_df = weather_df[["hour_timestamp", "temperature_2m"]].drop_duplicates(subset=["hour_timestamp"])

    weatherAndElec = pd.merge(meterElec_df, weatherSubset_df, on="hour_timestamp", how="left")
    allData_df = pd.merge(weatherAndElec, building_df, left_on="simscode", right_on="buildingnumber", how="left")

    # Group by building and hour, then aggregate the necessary columns
    clean_df = allData_df.groupby(["simscode", "hour_timestamp"]).agg({
        "hourly_est_kwh": "mean",
        "grossarea": "first",
        "constructiondate": "first",
        "temperature_2m": "mean"
    }).reset_index()

    clean_df["energyuseintensity"] = clean_df["hourly_est_kwh"] / clean_df["grossarea"]
    clean_df["hour_of_day"] = clean_df["hour_timestamp"].dt.hour
    clean_df["day_of_week"] = clean_df["hour_timestamp"].dt.dayofweek
    clean_df["is_weekend"] = clean_df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    clean_df = clean_df.dropna(subset=['temperature_2m', 'grossarea', 'constructiondate', 'energyuseintensity']).copy()
    clean_df['construction_year'] = pd.to_datetime(clean_df['constructiondate']).dt.year

    clean_df = remove_outliers(clean_df)
    print(f"Done!")

    clean_df.to_csv("processed/hourly_electricity_cleaned.csv", index=False)
    print(f"Cleaned data saved to \"processed/hourly_electricity_cleaned.csv!\"")

if __name__ == "__main__":
    run_data_prep()
