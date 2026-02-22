# Energy-Investment-Prioritization

Strategic Energy Investment Prioritization path for HackAI 2026

**By: Adam Hilty**

## Installation and Execution

### Prerequisites

* Python 3.10 or higher
* Project dependencies (installed via 'requirements.txt')
* Virtual environment (recommended)
* Relevant OSU data in 'data/' folder

### Execution Steps

1. Create and Activate Environment:

    python3 -m venv my_env

    source my_env/bin/activate

2. Install Dependencies

    pip install -r requirements.txt

3. Run the Pipeline (see Project Architecture for more information):

    python run_pipeline.py

4. Launch the Dashboard:

    streamlit run app.py

## HackAI 2026

Participated in Strategic Energy Investment Prioritization path. Project started 10:00am February 21, 2026 and ended 11:00 am February 22, 2026 when submissions were due.

## The Challenge

With hundreds of buildings and limited capital improvement budgets, facilities managers require a data-driven methodology to identify which assets will provide the highest return on investment for energy retrofits.

This project moves beyond simple "high energy use" lists by using Machine Learning to establish a fair performance baseline for every building, accounting for weather, time of day, and building size.

## Machine Learning Methodology

Instead of penalizing large buildings for their scale, I utilized an XGBoost Regression Model to establish a portfolio-wide performance baseline.

By calculating the deviation between a building's predicted energy use and its actual energy use, I generated three core investment signals:

Persistent Inefficiency: High average deviation over the observed window, suggesting systemic issues like degraded insulation.

Control Volatility: High standard deviation of errors, indicating erratic HVAC control systems or thermostat "hunting."

Nighttime Waste: Significant energy deviation between 12:00 AM and 6:00 AM, identifying buildings that fail to enter energy-saving modes.

## Project Architecture

This project is built as a modular, automated data pipeline to ensure reproducibility.

01_data_prep.py: Ingests raw meter and weather data and removes statistical outliers using the Interquartile Range (IQR) method.

02_model_training.py: Trains the XGBoost Regressor and generates hourly deviation datasets.

03_generate_perf_signals.py: Aggregates hourly data to produce the final building-level Investment Priority Scores.

run_pipeline.py: The orchestrator script that executes the entire backend sequence.

app.py: An interactive Streamlit dashboard for non-technical stakeholders to inspect findings.

## Author

**Adam Hilty**
GitHub [robbobthecorncob1](https://github.com/robbobthecorncob1) (School)


GitHub [bigred225](https://github.com/bigred225) (Personal)
