import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

clean_df = pd.read_csv("processed/hourly_electricity_cleaned.csv")

print("Generating plots...")
valid_buildings = clean_df['simscode'].dropna().unique()
sample_buildings = np.random.choice(valid_buildings, size=4, replace=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, building in enumerate(sample_buildings):
    oneBuilding = clean_df[clean_df['simscode'] == building]
    sns.regplot(
        data=oneBuilding, 
        x='temperature_2m', 
        y='energyuseintensity', 
        ax=axes[i], 
        scatter_kws={'alpha': 0.4, 'color': 'steelblue'},
        lowess=True,
        line_kws={'color': 'darkorange', 'linewidth': 2}
    )
    
    axes[i].set_title(f'Building SIMS Code: {building} - EUI vs Temp')
    axes[i].set_xlabel('Temperature (°F)')
    axes[i].set_ylabel('Energy Use Intensity (kWh/sqft)')
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()