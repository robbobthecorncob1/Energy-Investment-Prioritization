import pandas as pd

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

elecMeter_df = meter_df[meter_df["utility"] == "ELECTRICITY"].copy()
elecMeter_df["hourly_est_kwh"] = elecMeter_df["readingvalue"] * 4

elecMeter_df["hour_timestamp"] = elecMeter_df["readingtime"].dt.floor("h")
weather_df["hour_timestamp"] = weather_df["date"].dt.floor("h")
weatherAndElec = pd.merge(elecMeter_df, weather_df, on="hour_timestamp", how="left")
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

print(f"Done!")

clean_df.to_csv("processed/hourly_electricity_cleaned.csv", index=False)
print(f"Cleaned data saved to \"processed/hourly_electricity_cleaned.csv!\"")
