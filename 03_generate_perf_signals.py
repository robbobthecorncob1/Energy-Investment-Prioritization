import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def run_performance_signals():
    """
    Generates performance signals and saves them to the processed/ folder.
    
    Reads in the deviations created by the model training script, adds a check to see how the building performs at night,
    and then normalizes the mean EUI deviation, the volatitly of the EUI, and night waste.
    """
    buildingTemp_df = pd.read_csv("processed/building_deviations.csv")

    signals = buildingTemp_df.groupby('simscode').agg(
        mean_deviation=('eui_deviation', 'mean'),
        volatility=('eui_deviation', 'std'),
    ).reset_index()

    buildingTemp_df["is_night"] = buildingTemp_df["hour_of_day"].apply(lambda x: 1 if x < 6 else 0)
    night_stats = buildingTemp_df[buildingTemp_df['is_night'] == 1].groupby('simscode').agg(
        night_waste=('eui_deviation', 'mean')
    ).reset_index()

    building_scores = pd.merge(signals, night_stats, on='simscode', how='left')

    building_scores['night_waste'] = building_scores['night_waste'].fillna(0)

    print("Normalizing signals and generating final risk score...")
    building_scores[['norm_mean_dev', 'norm_volatility', 'norm_night_waste']] = MinMaxScaler().fit_transform(building_scores[['mean_deviation', 'volatility', 'night_waste']])

    # Weights: 50% Persistent Inefficiency, 25% Volatility, 25% Nighttime Waste
    building_scores['investment_priority_score'] = (
        (building_scores['norm_mean_dev'] * 0.50) + 
        (building_scores['norm_volatility'] * 0.25) + 
        (building_scores['norm_night_waste'] * 0.25)
    )

    final_ranking = building_scores.sort_values(by='investment_priority_score', ascending=False)

    final_ranking.to_csv("processed/final_building_rankings.csv", index=False)
    print("Scorecard saved to 'processed/final_building_rankings.csv'!")

if __name__ == "__main__":
    run_performance_signals()
    