import pandas as pd
import os

# Read in building, weather, and meter data
print(f"Reading in data...", end="")
building_df = pd.read_csv("data/building_metadata.csv")
weather_df = pd.read_csv("data/weather-sept-oct-2025.csv", parse_dates=["date"])
meter_sept = pd.read_csv("data/meter-data-sept-2025.csv", parse_dates=["readingtime"])
meter_oct = pd.read_csv("data/meter-data-oct-2025.csv", parse_dates=["readingtime"])
print(f"Success!")

print("Cleaning up data...", end="")
meter_df = pd.concat([meter_sept, meter_oct], ignore_index=True)
del meter_sept
del meter_oct

buildingAndMeter_df = pd.merge(meter_df, building_df, left_on="simscode", right_on="buildingnumber", how="left")

buildingAndMeter_df["join_date"] = buildingAndMeter_df["readingtime"].dt.normalize()
allData_df = pd.merge(buildingAndMeter_df, weather_df, left_on="join_date", right_on="date", how="left")
allData_df = allData_df.drop(columns=["join_date"])

elec_df = allData_df[allData_df["utility"] == "ELECTRICITY"].copy()
elec_df["hour_timestamp"] = elec_df["readingtime"].dt.floor("h")

# Group by building and hour, then aggregate the necessary columns
hourly_df = elec_df.groupby(["simscode", "hour_timestamp"]).agg({
    "readingvalue": "sum",
    "grossarea": "first",
    "constructiondate": "first",
    "temperature_2m": "mean"
}).reset_index()

hourly_df["energyUseIntensity"] = hourly_df["readingvalue"] / hourly_df["grossarea"]
print(f"Success!")

hourly_df.to_csv("processed/hourly_electricity_cleaned.csv", index=False)
print(f"Cleaned data saved to \"processed/hourly_electricity_cleaned.csv!\"")
