import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="OSU Smart Capital Planner", layout="wide")

@st.cache_data
def load_data():
    """
    Loads the ranked buildings and adds a building name table.
    Stores this data frame in cache.

    Returns:
        The data frame object.
    """
    df = pd.read_csv("processed/final_building_rankings.csv")
    meta_df = pd.read_csv("data/building_metadata.csv")

    return pd.merge(df, meta_df[['buildingnumber', 'buildingname']], left_on='simscode', right_on='buildingnumber', how='left')


df = load_data()

# -------- HEADER --------
st.title("🏛️ OSU Smart Capital Planner: AI-Driven Energy Investment")
st.markdown("""
**The Challenge:** If you have limited capital funding, which buildings should get energy retrofits first?  
**Our Approach:** We trained an XGBoost Machine Learning model on campus-wide data to establish a "fair baseline" for energy use based on weather, time, and building size. We then isolated buildings that consistently deviate from this baseline, ranking them by persistent inefficiency, erratic control systems, and nighttime waste.
""")
st.divider()

# -------- SECTION 1: THE LEADERBOARD --------
st.header("Top 10 Investment Priorities")
st.write("These buildings exhibit the highest financial risk based on our AI anomaly detection model.")

# Interactive table
top_10 = df.head(10).copy()
display_df = top_10[['buildingname', 'investment_priority_score', 'mean_deviation', 'volatility', 'night_waste']].copy()
display_df.columns = ['Building Name (#)', 'Overall Risk Score', 'Avg Hourly Waste (kWh/sqft)', 'Volatility (kWh/sqft)', 'Nighttime Waste (kWh/sqft)']
st.dataframe(display_df.style.background_gradient(cmap='Reds', subset=['Overall Risk Score']), hide_index=True)

st.divider()

# -------- SECTION 2: EXPLAINABILITY & DEEP DIVE --------
st.header("\U0001f50d Building Deep Dive: Why is this building flagged?")
st.write("Select a high-priority building to inspect its specific performance signals.")

# Dropdown to select a building from the Top 10
selected_building = st.selectbox("Select a Building:", top_10['buildingname'])
building_data = top_10[top_10['buildingname'] == selected_building].iloc[0]
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Signal Breakdown for Building {selected_building}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Inefficiency (kWh/sqft)", f"{building_data['mean_deviation']:.4f}")
    m2.metric("Volatility (kWh/sqft)", f"{building_data['volatility']:.4f}")
    m3.metric("Night Waste (kWh/sqft)", f"{building_data['night_waste']:.4f}")
    
    st.markdown(f"""
    * **Persistent Inefficiency:** Evaluates if the building constantly wastes energy regardless of weather.
    * **Control Volatility:** Measures how erratic the building's energy spikes are. High volatility often means failing HVAC controls.
    * **Nighttime Waste:** Captures energy wasted between midnight and 5 AM when the building should be empty.
    """)
    st.info("**AI Confidence & Uncertainty:** Rankings are derived from ~60 days of data. Buildings with high volatility carry a higher uncertainty band and require physical audits to confirm equipment faults.")

# Radar Chart
with col2:
    categories = ['Inefficiency', 'Volatility', 'Nighttime Waste']
    values = [
        building_data['norm_mean_dev'], 
        building_data['norm_volatility'], 
        building_data['norm_night_waste']
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=f'Building {selected_building}',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Normalized Risk Factors (0 to 1)"
    )
    st.plotly_chart(fig, width='stretch')

st.divider()

# -------- SECTION 3: ACTION FRAMING --------
st.header("Next Steps & Action Framing")
st.markdown("""
1. **Targeted Energy Audits:** Dispatch facilities teams to the top 3 buildings. Use this dashboard to guide their inspection (e.g., if Nighttime Waste is the primary driver, tell them to check scheduling overrides).
2. **Phased Capital Investment:** Allocate HVAC and insulation upgrade budgets starting with buildings exhibiting high *Persistent Inefficiency*.
3. **Future Data Integration:** With a longer time horizon (12+ months) and occupancy data, this framework could evolve to track post-retrofit savings and continuously re-rank the campus portfolio.
""")