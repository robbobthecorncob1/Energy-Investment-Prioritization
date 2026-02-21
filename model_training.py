import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def run_model_training():
    """
    Trains a XGBoost model to predicted expected energy use and calculates efficiency deviations.
    """

    clean_df = pd.read_csv("processed/hourly_electricity_cleaned.csv")

    # Defines the features and target of the XGBoost model
    features = [
        'temperature_2m', 
        'hour_of_day', 
        'day_of_week', 
        'is_weekend', 
        'grossarea', 
        'construction_year'
    ]
    clean_df['eui_scaled'] = clean_df['energyuseintensity'] * 1000  
    target = 'eui_scaled'

    X = clean_df[features]
    y = clean_df[target]

    # Train the model using 80% of the data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    # Test the model using the other 20% of the data 
    y_pred_test = xgb_model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    print(f"Model R-squared: {r2:.3f}")
    print(f"Mean Absolute Error (Scaled EUI): {mae:.3f}")

    # Calculate expected EUI and their deviations from the actual
    print("Calculating portfolio deviations...")
    clean_df['expected_eui_scaled'] = xgb_model.predict(X)
    clean_df['eui_deviation'] = (clean_df['eui_scaled'] - clean_df['expected_eui_scaled']) / 1000
    clean_df.to_csv("processed/building_deviations.csv", index=False)
    print("Deviations calculated and saved to \"processed/building_deviations.csv\"!")

if __name__ == "__main__":
    run_model_training()