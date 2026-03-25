import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import re
import tempfile
import os
from datetime import datetime

THRESHOLD = 80.0
SAFE_LIMIT = 60.0
WARNING_LIMIT = 80.0
ROLLING_WINDOW = 10
SPIKE_THRESHOLD_RATE = 5.0

st.set_page_config(
    page_title="Railway Locomotive Temperature Sensor Analysis",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Railway Locomotive Temperature Sensor Analysis")
st.markdown("Upload sensor data to analyze temperature patterns, detect threshold crossings, and generate reports.")

uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload an Excel file with columns: **Sensor Id**, **Temperature**, **time**")
    st.stop()

@st.cache_data(show_spinner="Reading Excel file...")
def load_excel(file_bytes):
    raw = pd.read_excel(io.BytesIO(file_bytes))
    legend = None
    try:
        legend = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Legend_Master")
    except (ValueError, Exception):
        pass
    return raw, legend

try:
    file_bytes = uploaded_file.getvalue()
    raw_df, legend_raw_cached = load_excel(file_bytes)
except ValueError as e:
    if "0 worksheets found" in str(e):
        st.error("This Excel file appears to have no worksheets. Please check that the file is not corrupted or password-protected, and try again with a valid .xlsx file.")
    else:
        st.error(f"Could not read the Excel file: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

required_columns = ["Sensor Id", "Temperature", "time"]
missing_cols = [c for c in required_columns if c not in raw_df.columns]
if missing_cols:
    st.error(f"Missing required columns: {', '.join(missing_cols)}")
    st.stop()

legend_df = None
has_legend = False
legend_warnings = []
if legend_raw_cached is not None:
    legend_raw = legend_raw_cached.copy()
    legend_required = ["Sl No", "Legend", "Sensor UID", "Description", "Threshold Temp"]
    legend_missing = [c for c in legend_required if c not in legend_raw.columns]
    if legend_missing:
        legend_warnings.append(f"Legend_Master sheet is missing columns: {', '.join(legend_missing)}. Proceeding without legend data.")
    else:
        legend_raw["Threshold Temp"] = pd.to_numeric(legend_raw["Threshold Temp"], errors="coerce")
        invalid_thresh = legend_raw["Threshold Temp"].isna().sum()
        if invalid_thresh > 0:
            legend_warnings.append(f"{invalid_thresh} row(s) in Legend_Master have non-numeric Threshold Temp values. These will use the default threshold ({THRESHOLD}°C).")
        dup_legends = legend_raw["Legend"].duplicated().sum()
        if dup_legends > 0:
            legend_warnings.append(f"Legend_Master has {dup_legends} duplicate Legend value(s). Keeping first occurrence.")
            legend_raw = legend_raw.drop_duplicates(subset="Legend", keep="first")
        dup_uids = legend_raw["Sensor UID"].dropna().duplicated().sum()
        if dup_uids > 0:
            legend_warnings.append(f"Legend_Master has {dup_uids} duplicate Sensor UID value(s). Keeping first occurrence.")
            legend_raw = legend_raw.drop_duplicates(subset="Sensor UID", keep="first")
        legend_df = legend_raw
        has_legend = True

if legend_warnings:
    for w in legend_warnings:
        st.warning(w)

if has_legend:
    st.success(f"Legend_Master sheet loaded successfully with {len(legend_df)} sensor entries.")


st.header("Data Quality Summary")
with st.expander("ℹ️ What is this section?"):
    st.markdown("""
**What it shows:** A summary of your uploaded data before analysis begins.

**How to read it:** Check for missing values, duplicates, and invalid entries. High numbers here may indicate data collection issues.

**Why it matters:** Poor data quality can lead to misleading results. This section helps you understand what was cleaned before analysis.

**What to look for:** Ideally, you want zero missing values and zero duplicates. A high number of invalid timestamps or temperatures may indicate sensor malfunctions.
""")

total_raw = len(raw_df)
duplicates = raw_df.duplicated().sum()
missing_sensor = raw_df["Sensor Id"].isna().sum()
missing_temp = raw_df["Temperature"].isna().sum()
missing_time = raw_df["time"].isna().sum()

raw_df["time"] = pd.to_datetime(raw_df["time"], errors="coerce")
invalid_timestamps = raw_df["time"].isna().sum() - missing_time

raw_df["Temperature"] = pd.to_numeric(raw_df["Temperature"], errors="coerce")
invalid_temps = raw_df["Temperature"].isna().sum() - missing_temp

valid_times = raw_df["time"].dropna()
min_ts = valid_times.min() if len(valid_times) > 0 else "N/A"
max_ts = valid_times.max() if len(valid_times) > 0 else "N/A"
unique_sensors = raw_df["Sensor Id"].nunique()

col1, col2, col3, col4, col4b = st.columns(5)
col1.metric("Total Raw Rows", f"{total_raw:,}")
col2.metric("Duplicate Rows", f"{duplicates:,}")
col3.metric("Unique Sensors", f"{unique_sensors}")
col4.metric("Invalid Timestamps", f"{invalid_timestamps}")
col4b.metric("Invalid Temperatures", f"{invalid_temps}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Missing Sensor Id", f"{missing_sensor}")
col6.metric("Missing Temperature", f"{missing_temp}")
col7.metric("Missing Time", f"{missing_time}")
col8.metric("Date Range", f"{str(min_ts)[:10]} to {str(max_ts)[:10]}" if min_ts != "N/A" else "N/A")

with st.expander("Records per Sensor"):
    sensor_counts = raw_df["Sensor Id"].value_counts().reset_index()
    sensor_counts.columns = ["Sensor Id", "Record Count"]
    st.dataframe(sensor_counts, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner="Cleaning and processing data...")
def clean_data(raw):
    d = raw.drop_duplicates()
    d = d.dropna(subset=["Sensor Id", "Temperature", "time"])
    d = d.sort_values("time").reset_index(drop=True)
    return d

df = clean_data(raw_df)

rows_after_clean = len(df)
st.caption(f"After cleaning: **{rows_after_clean:,}** rows (removed {total_raw - rows_after_clean:,} rows)")

if df.empty:
    st.warning("No valid data remaining after cleaning.")
    st.stop()

def derive_sensor_type(legend_str):
    s = str(legend_str).strip().upper()
    if s.startswith("TM"):
        return "TM"
    elif s.startswith("M"):
        return "MSU"
    elif s.startswith("A"):
        return "Axle"
    return "Other"

sensor_legend_map = {}
sensor_threshold_map = {}
sensor_type_map = {}
sensor_description_map = {}

if has_legend and legend_df is not None:
    uid_index = {}
    legend_index = {}
    for _, lrow in legend_df.iterrows():
        uid = lrow["Sensor UID"]
        leg = lrow["Legend"]
        if pd.notna(uid):
            uid_index[uid] = lrow
        if pd.notna(leg):
            legend_index[leg] = lrow

    for sid in df["Sensor Id"].unique():
        if sid in uid_index:
            row = uid_index[sid]
            sensor_legend_map[sid] = str(row["Legend"])
            sensor_description_map[sid] = str(row["Description"]) if pd.notna(row["Description"]) else ""
            sensor_threshold_map[sid] = float(row["Threshold Temp"]) if pd.notna(row["Threshold Temp"]) else THRESHOLD
            sensor_type_map[sid] = derive_sensor_type(row["Legend"])
        elif sid in legend_index:
            row = legend_index[sid]
            sensor_legend_map[sid] = str(sid)
            sensor_description_map[sid] = str(row["Description"]) if pd.notna(row["Description"]) else ""
            sensor_threshold_map[sid] = float(row["Threshold Temp"]) if pd.notna(row["Threshold Temp"]) else THRESHOLD
            sensor_type_map[sid] = derive_sensor_type(sid)
        else:
            sensor_legend_map[sid] = str(sid)
            sensor_threshold_map[sid] = THRESHOLD
            sensor_type_map[sid] = "Other"
            sensor_description_map[sid] = ""
else:
    for sid in df["Sensor Id"].unique():
        sensor_legend_map[sid] = str(sid)
        sensor_threshold_map[sid] = THRESHOLD
        sensor_type_map[sid] = "Other"
        sensor_description_map[sid] = ""

df["Display Label"] = df["Sensor Id"].map(sensor_legend_map)
df["Sensor Type"] = df["Sensor Id"].map(sensor_type_map)
df["Sensor Threshold"] = df["Sensor Id"].map(sensor_threshold_map)

def get_label(sensor_id):
    return sensor_legend_map.get(sensor_id, str(sensor_id))

all_sensors = sorted(df["Sensor Id"].unique().tolist())

if has_legend:
    all_sensor_types = sorted(set(sensor_type_map.values()))
    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 3])
else:
    filter_col1, filter_col2 = st.columns([2, 3])

with filter_col1:
    data_min_date = df["time"].min().date()
    data_max_date = df["time"].max().date()
    date_range = st.date_input(
        "Filter by Date Range",
        value=(data_min_date, data_max_date),
        min_value=data_min_date,
        max_value=data_max_date,
    )

if has_legend:
    with filter_col2:
        type_options = ["All"] + all_sensor_types
        selected_type = st.selectbox("Filter by Sensor Type", options=type_options, index=0)
    type_filtered_sensors = [s for s in all_sensors if selected_type == "All" or sensor_type_map.get(s) == selected_type]
    with filter_col3:
        selected_sensors = st.multiselect(
            "Filter by Sensor", options=type_filtered_sensors, default=type_filtered_sensors,
            format_func=lambda x: f"{get_label(x)} ({x})" if get_label(x) != str(x) else str(x)
        )
else:
    with filter_col2:
        selected_sensors = st.multiselect("Filter by Sensor", options=all_sensors, default=all_sensors)

if not selected_sensors:
    st.warning("Please select at least one sensor.")
    st.stop()

filtered_df = df[df["Sensor Id"].isin(selected_sensors)].copy()

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["time"].dt.date >= start_date) & (filtered_df["time"].dt.date <= end_date)
    ]
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    filtered_df = filtered_df[filtered_df["time"].dt.date == date_range[0]]

if filtered_df.empty:
    st.warning("No data found for the selected filters. Please adjust your date range or sensor selection.")
    st.stop()

filtered_min_ts = filtered_df["time"].min()
filtered_max_ts = filtered_df["time"].max()


def compute_crossing_alerts(data, threshold=THRESHOLD, per_sensor_thresholds=None):
    data = data.sort_values(["Sensor Id", "time"])
    if per_sensor_thresholds:
        thresholds = data["Sensor Id"].map(per_sensor_thresholds).fillna(threshold)
    else:
        thresholds = threshold
    above = data["Temperature"] > thresholds
    prev_above = above.groupby(data["Sensor Id"]).shift(1, fill_value=False)
    crossings = above & ~prev_above
    alert_rows = data[crossings].copy()
    if alert_rows.empty:
        return pd.DataFrame(columns=["Sensor Id", "Alert Time", "Temperature", "Threshold"])
    if per_sensor_thresholds:
        alert_rows["Threshold"] = alert_rows["Sensor Id"].map(per_sensor_thresholds).fillna(threshold)
    else:
        alert_rows["Threshold"] = threshold
    return alert_rows[["Sensor Id", "time", "Temperature", "Threshold"]].rename(columns={"time": "Alert Time"}).reset_index(drop=True)

alerts_df = compute_crossing_alerts(filtered_df, THRESHOLD, sensor_threshold_map if has_legend else None)
total_alerts = len(alerts_df)
total_records = len(filtered_df)
alert_pct = (total_alerts / total_records * 100) if total_records > 0 else 0


filtered_df = filtered_df.copy()
filtered_df["Rolling Avg"] = filtered_df.groupby("Sensor Id")["Temperature"].transform(
    lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
)

filtered_df["Temp Change"] = filtered_df.groupby("Sensor Id")["Temperature"].diff().fillna(0)
filtered_df["Is Spike"] = filtered_df["Temp Change"].abs() > SPIKE_THRESHOLD_RATE

spike_count = int(filtered_df["Is Spike"].sum())

sensor_stats = filtered_df.groupby("Sensor Id")["Temperature"].agg(["min", "max", "mean"]).reset_index()
sensor_stats.columns = ["Sensor Id", "Min Temp (°C)", "Max Temp (°C)", "Avg Temp (°C)"]
sensor_stats["Display Label"] = sensor_stats["Sensor Id"].map(sensor_legend_map)
sensor_stats["Sensor Type"] = sensor_stats["Sensor Id"].map(sensor_type_map)
sensor_stats["Sensor Threshold"] = sensor_stats["Sensor Id"].map(sensor_threshold_map)
sensor_stats["Avg Temp (°C)"] = sensor_stats["Avg Temp (°C)"].round(2)

if not alerts_df.empty:
    alert_counts = alerts_df.groupby("Sensor Id").size().reset_index(name="Alert Events")
    sensor_stats = sensor_stats.merge(alert_counts, on="Sensor Id", how="left")
    sensor_stats["Alert Events"] = sensor_stats["Alert Events"].fillna(0).astype(int)
else:
    sensor_stats["Alert Events"] = 0

spike_per_sensor = filtered_df[filtered_df["Is Spike"]].groupby("Sensor Id").size().reset_index(name="Spike Events")
sensor_stats = sensor_stats.merge(spike_per_sensor, on="Sensor Id", how="left")
sensor_stats["Spike Events"] = sensor_stats["Spike Events"].fillna(0).astype(int)

colors = px.colors.qualitative.Set2

MAX_POINTS_PER_TRACE = 2000

def downsample_sensor(sensor_data, max_points=MAX_POINTS_PER_TRACE):
    if len(sensor_data) <= max_points:
        return sensor_data
    step = max(1, len(sensor_data) // max_points)
    sampled = sensor_data.iloc[::step]
    idx_min = sensor_data["Temperature"].idxmin()
    idx_max = sensor_data["Temperature"].idxmax()
    extras = sensor_data.loc[[idx_min, idx_max]]
    result = pd.concat([sampled, extras]).drop_duplicates().sort_values("time")
    return result


st.header("Key Performance Indicators")
with st.expander("ℹ️ What are these KPIs?"):
    st.markdown("""
**What it shows:** High-level summary numbers for quick assessment of system health.

**How to read it:** These are the most important numbers at a glance. Green metrics are good; red metrics need attention.

**Why it matters:** Executives and operators can quickly assess whether the system is operating normally.

**Key terms:**
- **Alert Events** = Number of times a sensor's temperature crossed above the 80°C threshold (crossing-event logic)
- **Spike Events** = Number of sudden rapid temperature changes (>±5°C between consecutive readings) — an early warning signal
""")

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Sensors Monitored", f"{len(selected_sensors)}")
kpi2.metric("Total Records", f"{total_records:,}")
kpi3.metric("Alert Events", f"{total_alerts}")
kpi4.metric("Alert Rate", f"{alert_pct:.2f}%")
kpi5.metric("Spike Events", f"{spike_count}")
if has_legend:
    unique_thresh_vals = sorted(set(sensor_threshold_map.values()))
    if len(unique_thresh_vals) == 1:
        kpi6.metric("Threshold", f"{unique_thresh_vals[0]:.0f}°C")
    else:
        kpi6.metric("Thresholds", f"{min(unique_thresh_vals):.0f}–{max(unique_thresh_vals):.0f}°C")
    st.caption(f"Alert Events = per-sensor threshold crossings (thresholds: {', '.join(f'{v:.0f}°C' for v in unique_thresh_vals)}). Spike Events = rapid temperature changes (>±5°C between readings).")
else:
    kpi6.metric("Threshold", f"{THRESHOLD}°C")
    st.caption("Alert Events = threshold crossings (>80°C). Spike Events = rapid temperature changes (>±5°C between readings).")


st.header("Operational Insights")
with st.expander("ℹ️ What is this section?"):
    st.markdown("""
**What it shows:** A plain-language summary of what the data tells us about system health.

**How to read it:** Each bullet point highlights a key finding. These insights are designed for non-technical stakeholders.

**Why it matters:** Raw numbers alone don't tell the full story. This section translates the data into actionable observations.
""")

max_temp_sensor = sensor_stats.loc[sensor_stats["Max Temp (°C)"].idxmax()]
max_spike_sensor = sensor_stats.loc[sensor_stats["Spike Events"].idxmax()]

spike_df_times = filtered_df[filtered_df["Is Spike"]].copy()
time_clustering_text = ""
if not spike_df_times.empty:
    spike_df_times["Hour"] = spike_df_times["time"].dt.hour
    peak_hour = spike_df_times["Hour"].mode()
    if len(peak_hour) > 0:
        time_clustering_text = f"Spike events are most concentrated around hour {peak_hour.iloc[0]}:00, suggesting time-dependent operational patterns."

insights = []
if total_alerts == 0:
    insights.append("✅ **No threshold breaches observed** — All sensors remained within safe operating limits (below 80°C) throughout the monitoring period.")
else:
    insights.append(f"⚠️ **{total_alerts} threshold breach(es) detected** — {total_alerts} crossing events where temperature exceeded 80°C.")

if spike_count > 0:
    insights.append(f"📈 **Significant short-term variability detected** — {spike_count} spike events (rapid temperature changes >±5°C) were recorded. These are early warning signals even when temperatures stay below the threshold.")
else:
    insights.append("✅ **Stable temperature behavior** — No significant rapid temperature changes detected.")

insights.append(f"🌡️ **{get_label(max_temp_sensor['Sensor Id'])}** recorded the highest maximum temperature at **{max_temp_sensor['Max Temp (°C)']:.1f}°C**.")
insights.append(f"⚡ **{get_label(max_spike_sensor['Sensor Id'])}** showed the highest variability with **{int(max_spike_sensor['Spike Events'])} spike events**.")

if time_clustering_text:
    insights.append(f"🕐 {time_clustering_text}")

for insight in insights:
    st.markdown(insight)

st.divider()
st.markdown("**Understanding Alerts vs Spikes:**")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
**🔴 Alerts (Threshold Crossings)**
- Triggered when temperature rises **above 80°C**
- Uses crossing-event logic (only counted once per excursion)
- Indicates the system has exceeded safe limits
- Requires immediate attention
""")
with col_b:
    st.markdown("""
**🟡 Spikes (Rapid Changes)**
- Triggered when temperature changes **>±5°C** between consecutive readings
- Can occur at any temperature level
- Early warning signal of potential issues
- Indicates instability even within safe ranges
""")


st.header("Temperature vs Time")
st.markdown("_This section shows how temperature changes over time for each sensor. Zone shading indicates safe (green), warning (orange), and critical (red) operating regions._")
with st.expander("ℹ️ How to read this chart"):
    st.markdown("""
**What it shows:** Temperature readings over time for all selected sensors, with colored zone bands and a rolling average overlay.

**How to read it:** Each line represents a sensor. The dotted line is the smoothed rolling average. The red dashed line at 80°C is the alert threshold.

**Why it matters:** Helps identify trends, sudden changes, and whether sensors are approaching dangerous temperature levels.

**What to look for:** Lines approaching or entering the red zone, sudden jumps, and divergence between sensors.
""")

use_small_multiples = st.checkbox("Use small multiples (one chart per sensor)", value=len(selected_sensors) > 4)

if use_small_multiples:
    cols_per_row = min(3, len(selected_sensors))
    rows_needed = (len(selected_sensors) + cols_per_row - 1) // cols_per_row
    y_max = max(filtered_df["Temperature"].max(), 120)

    for row_idx in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            sensor_idx = row_idx * cols_per_row + col_idx
            if sensor_idx >= len(selected_sensors):
                break
            sensor_id = selected_sensors[sensor_idx]
            sensor_data = filtered_df[filtered_df["Sensor Id"] == sensor_id]
            color = colors[sensor_idx % len(colors)]

            fig_sm = go.Figure()
            fig_sm.add_hrect(y0=0, y1=SAFE_LIMIT, fillcolor="green", opacity=0.07, line_width=0)
            fig_sm.add_hrect(y0=SAFE_LIMIT, y1=WARNING_LIMIT, fillcolor="orange", opacity=0.07, line_width=0)
            fig_sm.add_hrect(y0=WARNING_LIMIT, y1=y_max * 1.1, fillcolor="red", opacity=0.07, line_width=0)

            s_thresh = sensor_threshold_map.get(sensor_id, THRESHOLD)
            ds = downsample_sensor(sensor_data)
            fig_sm.add_trace(go.Scatter(x=ds["time"], y=ds["Temperature"],
                                         mode="lines", name="Temp", line=dict(color=color, width=1.5)))
            fig_sm.add_trace(go.Scatter(x=ds["time"], y=ds["Rolling Avg"],
                                         mode="lines", name="Rolling Avg", line=dict(color=color, width=2.5, dash="dot")))
            fig_sm.add_hline(y=s_thresh, line_dash="dash", line_color="red", line_width=1.5)
            fig_sm.update_layout(title=f"{get_label(sensor_id)}", height=300, template="plotly_white",
                                  showlegend=False, margin=dict(l=40, r=10, t=40, b=30),
                                  yaxis=dict(range=[0, y_max * 1.1]))
            with cols[col_idx]:
                st.plotly_chart(fig_sm, use_container_width=True)
else:
    fig_temp = go.Figure()
    y_max = max(filtered_df["Temperature"].max(), 120)
    fig_temp.add_hrect(y0=0, y1=SAFE_LIMIT, fillcolor="green", opacity=0.07,
                       annotation_text="Safe Zone", annotation_position="top left", line_width=0)
    fig_temp.add_hrect(y0=SAFE_LIMIT, y1=WARNING_LIMIT, fillcolor="orange", opacity=0.07,
                       annotation_text="Warning Zone", annotation_position="top left", line_width=0)
    fig_temp.add_hrect(y0=WARNING_LIMIT, y1=y_max * 1.1, fillcolor="red", opacity=0.07,
                       annotation_text="Critical Zone", annotation_position="top left", line_width=0)
    for i, sensor_id in enumerate(selected_sensors):
        sensor_data = filtered_df[filtered_df["Sensor Id"] == sensor_id]
        ds = downsample_sensor(sensor_data)
        color = colors[i % len(colors)]
        label = get_label(sensor_id)
        fig_temp.add_trace(go.Scatter(x=ds["time"], y=ds["Temperature"],
                                       mode="lines", name=label, line=dict(color=color, width=1.5), opacity=0.7))
        fig_temp.add_trace(go.Scatter(x=ds["time"], y=ds["Rolling Avg"],
                                       mode="lines", name=f"{label} (Avg)",
                                       line=dict(color=color, width=2.5, dash="dot"), opacity=0.9))
    if has_legend:
        unique_thresholds = set(sensor_threshold_map.get(s, THRESHOLD) for s in selected_sensors)
        for t_val in sorted(unique_thresholds):
            fig_temp.add_hline(y=t_val, line_dash="dash", line_color="red", line_width=2,
                               annotation_text=f"Threshold ({t_val}°C)", annotation_position="top right")
    else:
        fig_temp.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", line_width=2,
                           annotation_text=f"Threshold ({THRESHOLD}°C)", annotation_position="top right")
    fig_temp.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)", height=500, template="plotly_white",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=60, r=30, t=60, b=60))
    st.plotly_chart(fig_temp, use_container_width=True)

st.caption("Use this chart to understand how temperature changes over time and identify patterns across sensors.")

temp_takeaway_parts = []
hottest_sensor = sensor_stats.loc[sensor_stats["Max Temp (°C)"].idxmax()]
temp_takeaway_parts.append(f"**{hottest_sensor['Display Label']}** recorded the highest temperature at **{hottest_sensor['Max Temp (°C)']:.1f}°C**.")
avg_temps = sensor_stats["Avg Temp (°C)"]
temp_spread = avg_temps.max() - avg_temps.min()
if temp_spread > 10:
    temp_takeaway_parts.append(f"Significant temperature spread of **{temp_spread:.1f}°C** between sensor averages — investigate sensors operating at different thermal levels.")
elif temp_spread < 3:
    temp_takeaway_parts.append(f"Sensors operate within a tight **{temp_spread:.1f}°C** average range — consistent thermal behavior across the fleet.")
sensors_above = sensor_stats[sensor_stats["Max Temp (°C)"] > THRESHOLD]
if len(sensors_above) > 0:
    temp_takeaway_parts.append(f"**{len(sensors_above)} of {len(sensor_stats)}** sensors exceeded the alert threshold during the monitoring period.")
st.info("📌 **Key Takeaway:** " + " ".join(temp_takeaway_parts))

st.header("Temperature Distribution")
st.markdown("_This section shows how temperature readings are distributed across all sensors, helping identify common operating ranges and outliers._")
with st.expander("ℹ️ How to read these charts"):
    st.markdown("""
**Histogram:** Shows how frequently each temperature range occurs. Peaks indicate the most common operating temperatures. The red line marks the alert threshold.

**Box Plot:** Shows the spread of temperatures per sensor. The box represents the middle 50% of readings. Whiskers show the full range. Dots beyond whiskers are outliers.

**Why it matters:** Helps identify whether sensors operate in similar ranges and spots unusual distributions.
""")

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    st.subheader("Histogram")
    fig_hist = px.histogram(filtered_df, x="Temperature", nbins=50, color_discrete_sequence=["#4C78A8"],
                             labels={"Temperature": "Temperature (°C)"})
    fig_hist.add_vline(x=THRESHOLD, line_dash="dash", line_color="red", line_width=2,
                       annotation_text=f"Threshold ({THRESHOLD}°C)")
    fig_hist.add_vline(x=SAFE_LIMIT, line_dash="dot", line_color="orange", line_width=1,
                       annotation_text=f"Warning ({SAFE_LIMIT}°C)")
    fig_hist.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("This histogram shows the frequency of temperature readings. Peaks indicate common operating temperatures.")

with dist_col2:
    st.subheader("Box Plot per Sensor")
    fig_box = px.box(filtered_df, x="Display Label", y="Temperature", color="Display Label",
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig_box.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", line_width=2)
    fig_box.update_layout(height=400, template="plotly_white", showlegend=False,
                           margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Compare the temperature range and spread across sensors. Wider boxes indicate more variability.")

dist_takeaway_parts = []
median_temps = filtered_df.groupby("Display Label")["Temperature"].median()
highest_median_sensor = median_temps.idxmax()
lowest_median_sensor = median_temps.idxmin()
dist_takeaway_parts.append(f"**{highest_median_sensor}** has the highest median temperature ({median_temps.max():.1f}°C) while **{lowest_median_sensor}** has the lowest ({median_temps.min():.1f}°C).")
std_devs = filtered_df.groupby("Display Label")["Temperature"].std()
most_variable = std_devs.idxmax()
dist_takeaway_parts.append(f"**{most_variable}** shows the most temperature variability (σ = {std_devs.max():.1f}°C) — may indicate inconsistent thermal load or cooling issues.")
st.info("📌 **Key Takeaway:** " + " ".join(dist_takeaway_parts))


st.header("Rate of Change Analysis")
st.markdown("_This section analyzes how quickly temperatures change between consecutive readings. Sudden spikes may indicate equipment issues._")
with st.expander("ℹ️ How to read this chart"):
    st.markdown("""
**What it shows:** The temperature difference between each consecutive pair of readings. Red diamonds mark spikes exceeding ±5°C.

**How to read it:** Most points should cluster near zero (stable operation). Points far from zero indicate rapid changes.

**Why it matters:** Rapid temperature changes can stress equipment even if the absolute temperature stays within safe limits. Spikes are early warning signals.

**What to look for:** Clusters of spikes at specific times or from specific sensors may indicate developing faults.
""")

fig_roc = go.Figure()
for i, sensor_id in enumerate(selected_sensors):
    sensor_data = filtered_df[filtered_df["Sensor Id"] == sensor_id]
    color = colors[i % len(colors)]
    label = get_label(sensor_id)
    normal = sensor_data[~sensor_data["Is Spike"]]
    spikes = sensor_data[sensor_data["Is Spike"]]
    ds_normal = downsample_sensor(normal) if len(normal) > MAX_POINTS_PER_TRACE else normal
    fig_roc.add_trace(go.Scatter(x=ds_normal["time"], y=ds_normal["Temp Change"], mode="markers", name=label,
                                  marker=dict(color=color, size=3, opacity=0.5)))
    if not spikes.empty:
        fig_roc.add_trace(go.Scatter(x=spikes["time"], y=spikes["Temp Change"], mode="markers",
                                      name=f"{label} (Spikes)",
                                      marker=dict(color="red", size=8, symbol="diamond", opacity=0.9)))
fig_roc.add_hline(y=SPIKE_THRESHOLD_RATE, line_dash="dot", line_color="red", line_width=1,
                  annotation_text=f"+{SPIKE_THRESHOLD_RATE}°C")
fig_roc.add_hline(y=-SPIKE_THRESHOLD_RATE, line_dash="dot", line_color="red", line_width=1,
                  annotation_text=f"-{SPIKE_THRESHOLD_RATE}°C")
fig_roc.update_layout(xaxis_title="Time", yaxis_title="Temperature Change (°C)", height=400, template="plotly_white",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                       margin=dict(l=60, r=30, t=60, b=60))
st.plotly_chart(fig_roc, use_container_width=True)
st.caption(f"Spike events detected (change > ±{SPIKE_THRESHOLD_RATE}°C between readings): **{spike_count}**. Red diamonds highlight these rapid changes.")

roc_takeaway_parts = []
if spike_count > 0:
    max_change = filtered_df["Temp Change"].abs().max()
    max_change_row = filtered_df.loc[filtered_df["Temp Change"].abs().idxmax()]
    roc_takeaway_parts.append(f"The largest single temperature change was **{max_change:.1f}°C** from sensor **{get_label(max_change_row['Sensor Id'])}**.")
    spike_pct = spike_count / total_records * 100
    roc_takeaway_parts.append(f"**{spike_count}** spike events detected across **{total_records:,}** readings ({spike_pct:.2f}% instability rate).")
    if spike_pct > 5:
        roc_takeaway_parts.append("High instability rate — investigate root causes of frequent rapid temperature changes.")
    elif spike_pct < 1:
        roc_takeaway_parts.append("Low instability rate — temperature changes are generally gradual and well-controlled.")
else:
    roc_takeaway_parts.append("No spike events detected — all temperature transitions were gradual and within normal rates.")
st.info("📌 **Key Takeaway:** " + " ".join(roc_takeaway_parts))


st.header("Spike Analysis")
st.markdown("_This section provides detailed breakdown of spike events — rapid temperature changes that serve as early warning signals._")
with st.expander("ℹ️ What is spike analysis?"):
    st.markdown("""
**What it shows:** When and where rapid temperature changes occurred, broken down by sensor and over time.

**Why it matters:** Even if temperatures stay below the threshold, frequent spikes indicate instability and potential developing issues.

**What to look for:** Sensors with many spikes may need maintenance. Time patterns may indicate operational causes.
""")

spike_col1, spike_col2 = st.columns(2)

with spike_col1:
    st.subheader("Spike Count per Sensor")
    spike_sensor_df = sensor_stats.copy()
    fig_spike_sensor = px.bar(spike_sensor_df, x="Display Label", y="Spike Events",
                               color="Spike Events", color_continuous_scale="YlOrRd",
                               labels={"Spike Events": "Number of Spikes", "Display Label": "Sensor"})
    fig_spike_sensor.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40),
                                    xaxis_type="category")
    st.plotly_chart(fig_spike_sensor, use_container_width=True)
    st.caption("This chart shows which sensors experience the most rapid temperature changes.")

with spike_col2:
    st.subheader("Spike Events Over Time")
    if not spike_df_times.empty:
        spike_daily = spike_df_times.copy()
        spike_daily["Date"] = spike_daily["time"].dt.date
        spike_by_date = spike_daily.groupby("Date").size().reset_index(name="Spike Events")
        fig_spike_time = px.bar(spike_by_date, x="Date", y="Spike Events",
                                 color_discrete_sequence=["#FF8C00"],
                                 labels={"Date": "Date", "Spike Events": "Number of Spikes"})
        fig_spike_time.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_spike_time, use_container_width=True)
        st.caption("This chart shows how spike events are distributed over time. Clusters may indicate operational patterns.")
    else:
        st.info("No spike events detected.")

spike_takeaway_parts = []
if spike_count > 0:
    spikes_per_sensor = sensor_stats[["Display Label", "Spike Events"]].sort_values("Spike Events", ascending=False)
    top_spike_sensor = spikes_per_sensor.iloc[0]
    spike_takeaway_parts.append(f"**{top_spike_sensor['Display Label']}** leads with **{int(top_spike_sensor['Spike Events'])}** spike events.")
    sensors_with_spikes = int((spikes_per_sensor["Spike Events"] > 0).sum())
    spike_takeaway_parts.append(f"**{sensors_with_spikes} of {len(spikes_per_sensor)}** sensors experienced at least one spike event.")
    if not spike_df_times.empty:
        spike_df_times_copy = spike_df_times.copy()
        spike_df_times_copy["Date"] = spike_df_times_copy["time"].dt.date
        worst_day = spike_df_times_copy.groupby("Date").size().idxmax()
        worst_day_count = spike_df_times_copy.groupby("Date").size().max()
        spike_takeaway_parts.append(f"Peak spike activity occurred on **{worst_day}** with **{worst_day_count}** events.")
else:
    spike_takeaway_parts.append("No spike events detected — all temperature transitions remained within normal rates across all sensors.")
st.info("📌 **Key Takeaway:** " + " ".join(spike_takeaway_parts))


st.header("Alert Summary")
st.markdown("_This section summarizes threshold crossing events — instances where sensors exceeded the 80°C safety limit._")
with st.expander("ℹ️ How to read the alert summary"):
    st.markdown("""
**What it shows:** The number and distribution of alert events (temperature crossing above 80°C).

**How to read it:** Alert events use crossing-event logic — only counted when temperature transitions from safe to above threshold, not for every reading above 80°C.

**Why it matters:** Alerts indicate that equipment has exceeded safe operating limits and may require inspection or maintenance.
""")

alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
alert_col1.metric("Total Alert Events", f"{total_alerts}")
alert_col2.metric("Total Records", f"{total_records:,}")
alert_col3.metric("Alert Event %", f"{alert_pct:.2f}%")
alert_col4.metric("Sensors Monitored", f"{len(selected_sensors)}")


alert_chart_col1, alert_chart_col2 = st.columns(2)

with alert_chart_col1:
    st.subheader("Alerts per Sensor")
    if not alerts_df.empty:
        alerts_per_sensor = alerts_df.groupby("Sensor Id").size().reset_index(name="Alert Events")
        alerts_per_sensor["Display Label"] = alerts_per_sensor["Sensor Id"].map(sensor_legend_map)
        fig_alerts_sensor = px.bar(alerts_per_sensor, x="Display Label", y="Alert Events",
                                    color="Alert Events", color_continuous_scale="Reds",
                                    labels={"Display Label": "Sensor"})
        fig_alerts_sensor.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40),
                                         xaxis_type="category")
        st.plotly_chart(fig_alerts_sensor, use_container_width=True)
        st.caption("This chart highlights which sensors trigger the most threshold crossing alerts.")
    else:
        st.info("No alert events detected — all sensors remained within safe limits.")

with alert_chart_col2:
    st.subheader("Daily Alert Trend")
    if not alerts_df.empty:
        alerts_df["Alert Date"] = alerts_df["Alert Time"].dt.date
        daily_alerts = alerts_df.groupby("Alert Date").size().reset_index(name="Alert Events")
        fig_daily = px.bar(daily_alerts, x="Alert Date", y="Alert Events", color_discrete_sequence=["#E45756"])
        fig_daily.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_daily, use_container_width=True)
        st.caption("This chart shows how alert events are distributed over days. Clusters may indicate recurring issues.")
    else:
        st.info("No alert events detected.")


st.subheader("Monthly Alert Trend")
with st.expander("ℹ️ How to read this chart"):
    st.markdown("""
**What it shows:** Alert events aggregated by month to reveal long-term trends.

**Why it matters:** Increasing monthly alerts may indicate deteriorating equipment condition over time.
""")
if not alerts_df.empty:
    alerts_df["Alert Month"] = alerts_df["Alert Time"].dt.to_period("M").astype(str)
    monthly_alerts = alerts_df.groupby("Alert Month").size().reset_index(name="Alert Events")
    fig_monthly = px.bar(monthly_alerts, x="Alert Month", y="Alert Events", color_discrete_sequence=["#FF6B6B"],
                          labels={"Alert Month": "Month", "Alert Events": "Alert Events"})
    fig_monthly.update_layout(height=350, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_monthly, use_container_width=True)
    st.caption("Use this chart to identify seasonal or long-term trends in alert frequency.")
else:
    st.info("No alert events to display monthly trend.")

alert_takeaway_parts = []
if total_alerts > 0:
    alerts_per_sensor_df = sensor_stats[["Display Label", "Alert Events"]].sort_values("Alert Events", ascending=False)
    top_alert_sensor = alerts_per_sensor_df.iloc[0]
    alert_takeaway_parts.append(f"**{int(top_alert_sensor['Alert Events'])}** alert events detected across **{len(alerts_per_sensor_df[alerts_per_sensor_df['Alert Events'] > 0])}** sensors.")
    alert_takeaway_parts.append(f"**{top_alert_sensor['Display Label']}** has the most alerts with **{int(top_alert_sensor['Alert Events'])}** threshold crossing events.")
    if alert_pct > 5:
        alert_takeaway_parts.append("Alert rate exceeds 5% — systematic thermal issues require investigation.")
    elif alert_pct < 1:
        alert_takeaway_parts.append("Alert rate below 1% — threshold crossings are infrequent and may be transient.")
else:
    alert_takeaway_parts.append("No alert events detected — all sensors remained within safe operating limits throughout the monitoring period.")
st.info("📌 **Key Takeaway:** " + " ".join(alert_takeaway_parts))


st.header("Sensor Rankings")
st.markdown("_Compare sensors to identify which ones require the most attention based on alerts, spikes, and peak temperatures._")
with st.expander("ℹ️ How to read sensor rankings"):
    st.markdown("""
**What it shows:** Sensors ranked by different risk factors — alert count, spike count, and maximum temperature.

**Why it matters:** Helps prioritize maintenance and inspection efforts on the most problematic sensors.

**What to look for:** Sensors consistently ranking high across multiple criteria may need immediate attention.
""")

rank_col1, rank_col2, rank_col3 = st.columns(3)

with rank_col1:
    st.subheader("By Alert Count")
    alert_rank = sensor_stats[["Display Label", "Alert Events"]].sort_values("Alert Events", ascending=False).reset_index(drop=True)
    alert_rank.index = alert_rank.index + 1
    alert_rank.index.name = "Rank"
    st.dataframe(alert_rank, use_container_width=True)

with rank_col2:
    st.subheader("By Spike Count")
    spike_rank = sensor_stats[["Display Label", "Spike Events"]].sort_values("Spike Events", ascending=False).reset_index(drop=True)
    spike_rank.index = spike_rank.index + 1
    spike_rank.index.name = "Rank"
    st.dataframe(spike_rank, use_container_width=True)

with rank_col3:
    st.subheader("By Max Temperature")
    max_temp_rank = sensor_stats[["Display Label", "Max Temp (°C)"]].sort_values("Max Temp (°C)", ascending=False).reset_index(drop=True)
    max_temp_rank.index = max_temp_rank.index + 1
    max_temp_rank.index.name = "Rank"
    st.dataframe(max_temp_rank, use_container_width=True)

rank_takeaway_parts = []
alert_top = sensor_stats.loc[sensor_stats["Alert Events"].idxmax()]
spike_top = sensor_stats.loc[sensor_stats["Spike Events"].idxmax()]
temp_top = sensor_stats.loc[sensor_stats["Max Temp (°C)"].idxmax()]
if alert_top["Display Label"] == spike_top["Display Label"] == temp_top["Display Label"]:
    rank_takeaway_parts.append(f"**{alert_top['Display Label']}** ranks #1 across all three categories (alerts, spikes, and max temperature) — this sensor requires priority attention.")
else:
    top_sensors = set([alert_top["Display Label"], spike_top["Display Label"], temp_top["Display Label"]])
    rank_takeaway_parts.append(f"Top-ranked sensors: **{alert_top['Display Label']}** (most alerts), **{spike_top['Display Label']}** (most spikes), **{temp_top['Display Label']}** (highest peak temperature).")
    if len(top_sensors) > 2:
        rank_takeaway_parts.append("Risk is spread across multiple sensors — review each for different failure modes.")
st.info("📌 **Key Takeaway:** " + " ".join(rank_takeaway_parts))


st.header("Sensor Summary")
st.markdown("_Complete summary table showing all key metrics for each sensor in one view._")
with st.expander("ℹ️ How to read the sensor summary"):
    st.markdown("""
**What it shows:** A table with min, max, average temperature, alert count, and spike count for each sensor.

**How to read it:** Higher max temperatures and more alerts/spikes indicate sensors that need attention.

**Why it matters:** Provides a single reference point for comparing all sensors side by side.
""")
sensor_summary_display = sensor_stats.copy()
if has_legend:
    display_cols = ["Display Label", "Sensor Type", "Min Temp (°C)", "Max Temp (°C)", "Avg Temp (°C)", "Sensor Threshold", "Alert Events", "Spike Events"]
    display_cols = [c for c in display_cols if c in sensor_summary_display.columns]
else:
    display_cols = ["Sensor Id", "Min Temp (°C)", "Max Temp (°C)", "Avg Temp (°C)", "Alert Events", "Spike Events"]
st.dataframe(sensor_summary_display[display_cols], use_container_width=True, hide_index=True)
st.caption("This table provides a comprehensive comparison of all monitored sensors.")

summary_takeaway_parts = []
total_sensor_count = len(sensor_stats)
problematic = sensor_stats[(sensor_stats["Alert Events"] > 0) | (sensor_stats["Spike Events"] > 0)]
if len(problematic) > 0:
    summary_takeaway_parts.append(f"**{len(problematic)} of {total_sensor_count}** sensors have recorded at least one alert or spike event.")
    clean_sensors = total_sensor_count - len(problematic)
    if clean_sensors > 0:
        summary_takeaway_parts.append(f"**{clean_sensors}** sensors maintained clean records with no alerts or spikes throughout the monitoring period.")
else:
    summary_takeaway_parts.append(f"All **{total_sensor_count}** sensors maintained clean records — no alerts or spikes detected across the entire monitoring period.")
st.info("📌 **Key Takeaway:** " + " ".join(summary_takeaway_parts))


st.header("Temperature Heatmap")
st.markdown("_This heatmap provides a visual overview of temperature patterns across sensors and time. Higher intensity indicates higher temperature concentration._")
with st.expander("ℹ️ How to read the heatmap"):
    st.markdown("""
**What it shows:** Average temperature for each sensor at each time period, displayed as a color grid.

**How to read it:** Warmer colors (red/orange) indicate higher temperatures. Cooler colors (green/yellow) indicate lower temperatures.

**Why it matters:** Quickly spot which sensors run hot and when. Patterns across time may reveal operational cycles.

**What to look for:** Horizontal bands of warm colors indicate consistently hot sensors. Vertical bands indicate time periods when all sensors ran hot.
""")

heatmap_df = filtered_df.copy()
time_range = (filtered_max_ts - filtered_min_ts).total_seconds() / 3600
if time_range > 48:
    heatmap_df["Time Bucket"] = heatmap_df["time"].dt.floor("2h")
else:
    heatmap_df["Time Bucket"] = heatmap_df["time"].dt.floor("h")

heatmap_df["Display Label"] = heatmap_df["Sensor Id"].map(sensor_legend_map)
heatmap_pivot = heatmap_df.pivot_table(values="Temperature", index="Display Label", columns="Time Bucket", aggfunc="mean")

if not heatmap_pivot.empty:
    fig_heatmap = px.imshow(
        heatmap_pivot.values,
        x=[str(c)[:16] for c in heatmap_pivot.columns],
        y=[str(s) for s in heatmap_pivot.index],
        color_continuous_scale="RdYlGn_r",
        labels=dict(x="Time", y="Sensor Id", color="Temp (°C)"),
        aspect="auto",
    )
    fig_heatmap.update_layout(
        height=max(300, len(heatmap_pivot.index) * 50 + 120),
        template="plotly_white",
        margin=dict(l=80, r=30, t=40, b=80),
        xaxis=dict(tickangle=-45),
        coloraxis_colorbar=dict(title="Temp (°C)", len=0.8),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption("Higher intensity (red/orange) indicates higher temperature concentration. Green areas indicate cooler, safer operating conditions.")

    heatmap_takeaway_parts = []
    heatmap_max_sensor = heatmap_pivot.max(axis=1).idxmax()
    heatmap_max_time = heatmap_pivot.max(axis=0).idxmax()
    heatmap_takeaway_parts.append(f"Hottest sensor across all time periods: **{heatmap_max_sensor}** (peak avg: {heatmap_pivot.max(axis=1).max():.1f}°C).")
    heatmap_takeaway_parts.append(f"Hottest time period: **{str(heatmap_max_time)[:16]}** (peak avg: {heatmap_pivot.max(axis=0).max():.1f}°C across sensors).")
    row_range = heatmap_pivot.max(axis=1) - heatmap_pivot.min(axis=1)
    most_variable_hm = row_range.idxmax()
    if row_range.max() > 5:
        heatmap_takeaway_parts.append(f"**{most_variable_hm}** shows the widest temperature swing ({row_range.max():.1f}°C range) across time periods.")
    st.info("📌 **Key Takeaway:** " + " ".join(heatmap_takeaway_parts))
else:
    st.info("Not enough data to generate heatmap.")


st.header("Temporal Patterns Analysis")
st.markdown("_This section analyzes WHEN temperature behavior changes — by time of day, day of week, and month — helping identify operational patterns and scheduling insights._")
with st.expander("ℹ️ What is temporal analysis?"):
    st.markdown("""
**What it shows:** Temperature and spike patterns broken down by hour of day, day of week, and month.

**How to read it:** Look for recurring patterns — certain hours or days that consistently show higher temperatures or more spike events.

**Why it matters:** Understanding WHEN issues occur helps with scheduling maintenance, adjusting operations, and identifying root causes tied to time-based factors (shift changes, load cycles, environmental conditions).

**What to look for:** Peak hours with high temperatures or spikes, days with more variability, and seasonal trends across months.
""")

filtered_df["Hour"] = filtered_df["time"].dt.hour
filtered_df["DayOfWeek"] = filtered_df["time"].dt.day_name()
filtered_df["Month"] = filtered_df["time"].dt.to_period("M").astype(str)
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.subheader("Time-of-Day Analysis")
tod_col1, tod_col2 = st.columns(2)

with tod_col1:
    hourly_avg = filtered_df.groupby("Hour")["Temperature"].mean().reset_index()
    hourly_avg.columns = ["Hour", "Avg Temperature (°C)"]
    fig_hourly_temp = px.bar(hourly_avg, x="Hour", y="Avg Temperature (°C)",
                              color="Avg Temperature (°C)", color_continuous_scale="RdYlGn_r",
                              labels={"Hour": "Hour of Day (0–23)", "Avg Temperature (°C)": "Avg Temp (°C)"})
    fig_hourly_temp.update_layout(height=400, template="plotly_white",
                                   xaxis=dict(dtick=1), margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_hourly_temp, use_container_width=True)
    st.caption("Average temperature by hour of day. Higher bars indicate hotter operating hours.")

with tod_col2:
    hourly_spikes = filtered_df[filtered_df["Is Spike"]].groupby("Hour").size().reset_index(name="Spike Count")
    all_hours = pd.DataFrame({"Hour": range(24)})
    hourly_spikes = all_hours.merge(hourly_spikes, on="Hour", how="left").fillna(0)
    hourly_spikes["Spike Count"] = hourly_spikes["Spike Count"].astype(int)
    fig_hourly_spikes = px.bar(hourly_spikes, x="Hour", y="Spike Count",
                                color="Spike Count", color_continuous_scale="YlOrRd",
                                labels={"Hour": "Hour of Day (0–23)", "Spike Count": "Spikes"})
    fig_hourly_spikes.update_layout(height=400, template="plotly_white",
                                     xaxis=dict(dtick=1), margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_hourly_spikes, use_container_width=True)
    st.caption("Spike events by hour of day. Clusters indicate time-dependent instability.")

st.subheader("Day-of-Week Analysis")
dow_col1, dow_col2 = st.columns(2)

with dow_col1:
    daily_avg = filtered_df.groupby("DayOfWeek")["Temperature"].mean().reset_index()
    daily_avg.columns = ["Day", "Avg Temperature (°C)"]
    daily_avg["Day"] = pd.Categorical(daily_avg["Day"], categories=day_order, ordered=True)
    daily_avg = daily_avg.sort_values("Day")
    fig_dow_temp = px.bar(daily_avg, x="Day", y="Avg Temperature (°C)",
                           color="Avg Temperature (°C)", color_continuous_scale="RdYlGn_r",
                           labels={"Day": "Day of Week", "Avg Temperature (°C)": "Avg Temp (°C)"})
    fig_dow_temp.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_dow_temp, use_container_width=True)
    st.caption("Average temperature by day of week. Compare weekday vs weekend patterns.")

with dow_col2:
    dow_spikes = filtered_df[filtered_df["Is Spike"]].groupby("DayOfWeek").size().reset_index(name="Spike Count")
    all_days = pd.DataFrame({"DayOfWeek": day_order})
    dow_spikes = all_days.merge(dow_spikes, on="DayOfWeek", how="left").fillna(0)
    dow_spikes["Spike Count"] = dow_spikes["Spike Count"].astype(int)
    dow_spikes["DayOfWeek"] = pd.Categorical(dow_spikes["DayOfWeek"], categories=day_order, ordered=True)
    dow_spikes = dow_spikes.sort_values("DayOfWeek")
    fig_dow_spikes = px.bar(dow_spikes, x="DayOfWeek", y="Spike Count",
                             color="Spike Count", color_continuous_scale="YlOrRd",
                             labels={"DayOfWeek": "Day of Week", "Spike Count": "Spikes"})
    fig_dow_spikes.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_dow_spikes, use_container_width=True)
    st.caption("Spike events by day of week. Higher counts on certain days may indicate operational patterns.")

st.subheader("Monthly Trends")
monthly_col1, monthly_col2, monthly_col3 = st.columns(3)

with monthly_col1:
    monthly_avg_temp = filtered_df.groupby("Month")["Temperature"].mean().reset_index()
    monthly_avg_temp.columns = ["Month", "Avg Temperature (°C)"]
    fig_monthly_temp = px.bar(monthly_avg_temp, x="Month", y="Avg Temperature (°C)",
                               color="Avg Temperature (°C)", color_continuous_scale="RdYlGn_r")
    fig_monthly_temp.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_monthly_temp, use_container_width=True)
    st.caption("Average temperature per month. Look for seasonal patterns or long-term trends.")

with monthly_col2:
    monthly_spikes_data = filtered_df[filtered_df["Is Spike"]].copy()
    monthly_spikes_data["Month"] = monthly_spikes_data["time"].dt.to_period("M").astype(str)
    if not monthly_spikes_data.empty:
        monthly_spike_count = monthly_spikes_data.groupby("Month").size().reset_index(name="Spike Count")
    else:
        monthly_spike_count = pd.DataFrame({"Month": [], "Spike Count": []})
    fig_monthly_spikes = px.bar(monthly_spike_count, x="Month", y="Spike Count",
                                 color="Spike Count", color_continuous_scale="YlOrRd")
    fig_monthly_spikes.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_monthly_spikes, use_container_width=True)
    st.caption("Spike events per month. Increasing trend may indicate equipment degradation.")

with monthly_col3:
    if not alerts_df.empty:
        alerts_monthly = alerts_df.copy()
        alerts_monthly["Month"] = alerts_monthly["Alert Time"].dt.to_period("M").astype(str)
        monthly_alert_count = alerts_monthly.groupby("Month").size().reset_index(name="Alert Count")
    else:
        monthly_alert_count = pd.DataFrame({"Month": [], "Alert Count": []})
    fig_monthly_alerts = px.bar(monthly_alert_count, x="Month", y="Alert Count",
                                 color_discrete_sequence=["#E45756"])
    fig_monthly_alerts.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_monthly_alerts, use_container_width=True)
    st.caption("Alert events per month. Track threshold breaches over time.")

st.subheader("Time-of-Day Heatmap")
with st.expander("ℹ️ How to read the time-of-day heatmap"):
    st.markdown("""
**What it shows:** Average temperature for each sensor at each hour of the day.

**How to read it:** Each cell represents a sensor (Y-axis) at a specific hour (X-axis). Warmer colors mean higher temperatures.

**Why it matters:** Quickly reveals which sensors run hot at which times, helping identify load cycles, shift patterns, or environmental influences.

**What to look for:** Horizontal bands of warm colors indicate sensors that are consistently hot. Vertical bands indicate hours when all sensors run hot together.
""")

tod_heatmap = filtered_df.pivot_table(values="Temperature", index="Display Label", columns="Hour", aggfunc="mean")
if not tod_heatmap.empty:
    fig_tod_heatmap = px.imshow(
        tod_heatmap.values,
        x=[f"{h}:00" for h in tod_heatmap.columns],
        y=[str(s) for s in tod_heatmap.index],
        color_continuous_scale="RdYlGn_r",
        labels=dict(x="Hour of Day", y="Sensor Id", color="Avg Temp (°C)"),
        aspect="auto",
    )
    fig_tod_heatmap.update_layout(
        height=max(300, len(tod_heatmap.index) * 50 + 120),
        template="plotly_white",
        margin=dict(l=80, r=30, t=40, b=80),
        coloraxis_colorbar=dict(title="Avg Temp (°C)", len=0.8),
    )
    st.plotly_chart(fig_tod_heatmap, use_container_width=True)
    st.caption("Average temperature by sensor and hour. Warmer colors (red/orange) indicate higher average temperatures at that hour.")
else:
    st.info("Not enough data to generate time-of-day heatmap.")

peak_hour_temp = hourly_avg.loc[hourly_avg["Avg Temperature (°C)"].idxmax()]
coolest_hour_temp = hourly_avg.loc[hourly_avg["Avg Temperature (°C)"].idxmin()]
st.markdown("**Key Temporal Insights:**")
st.markdown(f"- 🔥 **Peak hour:** {int(peak_hour_temp['Hour'])}:00 with an average of {peak_hour_temp['Avg Temperature (°C)']:.1f}°C")
st.markdown(f"- ❄️ **Coolest hour:** {int(coolest_hour_temp['Hour'])}:00 with an average of {coolest_hour_temp['Avg Temperature (°C)']:.1f}°C")
if not hourly_spikes[hourly_spikes["Spike Count"] > 0].empty:
    peak_spike_hour = hourly_spikes.loc[hourly_spikes["Spike Count"].idxmax()]
    st.markdown(f"- ⚡ **Most spike-prone hour:** {int(peak_spike_hour['Hour'])}:00 with {int(peak_spike_hour['Spike Count'])} spike events")
if not daily_avg.empty:
    peak_day = daily_avg.loc[daily_avg["Avg Temperature (°C)"].idxmax()]
    st.markdown(f"- 📅 **Hottest day of week:** {peak_day['Day']} with an average of {peak_day['Avg Temperature (°C)']:.1f}°C")


if has_legend:
    st.header("Threshold Analysis")
    st.markdown("_Per-sensor threshold analysis using uploaded threshold values from the Legend_Master sheet._")
    with st.expander("ℹ️ What is threshold analysis?"):
        st.markdown("""
**What it shows:** Detailed threshold breach analysis using per-sensor threshold values from the Legend_Master sheet.

**How to read it:** Each sensor has its own threshold. This section shows which sensors breach their specific thresholds and by how much.

**Why it matters:** Different sensor types may have different safe operating limits. Per-sensor thresholds provide more accurate breach detection.
""")

    thresh_col1, thresh_col2 = st.columns(2)
    with thresh_col1:
        st.subheader("Breach Count by Sensor")
        breach_data = []
        for sid in selected_sensors:
            s_data = filtered_df[filtered_df["Sensor Id"] == sid]
            s_thresh = sensor_threshold_map.get(sid, THRESHOLD)
            breach_count = int((s_data["Temperature"] > s_thresh).sum())
            breach_data.append({
                "Sensor": get_label(sid),
                "Threshold (°C)": s_thresh,
                "Breach Count": breach_count,
                "Max Temp (°C)": round(s_data["Temperature"].max(), 1),
                "Margin (°C)": round(s_data["Temperature"].max() - s_thresh, 1),
            })
        breach_df = pd.DataFrame(breach_data)
        fig_breach = px.bar(breach_df, x="Sensor", y="Breach Count",
                             color="Breach Count", color_continuous_scale="Reds",
                             labels={"Sensor": "Sensor", "Breach Count": "Readings Above Threshold"})
        fig_breach.update_layout(height=400, template="plotly_white", xaxis_type="category",
                                  margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_breach, use_container_width=True)

    with thresh_col2:
        st.subheader("Max Temperature vs Threshold")
        fig_max_vs_thresh = go.Figure()
        sensors_sorted = sorted(selected_sensors, key=lambda s: sensor_stats[sensor_stats["Sensor Id"]==s]["Max Temp (°C)"].values[0], reverse=True)
        labels_sorted = [get_label(s) for s in sensors_sorted]
        max_temps = [sensor_stats[sensor_stats["Sensor Id"]==s]["Max Temp (°C)"].values[0] for s in sensors_sorted]
        thresholds = [sensor_threshold_map.get(s, THRESHOLD) for s in sensors_sorted]
        fig_max_vs_thresh.add_trace(go.Bar(x=labels_sorted, y=max_temps, name="Max Temperature",
                                            marker_color="#3498DB"))
        fig_max_vs_thresh.add_trace(go.Scatter(x=labels_sorted, y=thresholds, name="Threshold",
                                                mode="markers+lines", line=dict(color="red", width=2, dash="dash"),
                                                marker=dict(color="red", size=10)))
        fig_max_vs_thresh.update_layout(height=400, template="plotly_white", barmode="group",
                                          yaxis_title="Temperature (°C)",
                                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                          margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig_max_vs_thresh, use_container_width=True)

    st.subheader("Top Recurring Threshold Breaches")
    recurring_data = []
    for sid in selected_sensors:
        s_data = filtered_df[filtered_df["Sensor Id"] == sid].sort_values("time")
        s_thresh = sensor_threshold_map.get(sid, THRESHOLD)
        above_mask = s_data["Temperature"] > s_thresh
        crossings = int((above_mask & ~above_mask.shift(1, fill_value=False)).sum())
        recurring_data.append({
            "Sensor": get_label(sid),
            "Threshold (°C)": s_thresh,
            "Crossing Events": crossings,
            "Readings Above": int(above_mask.sum()),
            "% Above": round(above_mask.mean() * 100, 2),
        })
    recurring_df = pd.DataFrame(recurring_data).sort_values("Crossing Events", ascending=False)
    st.dataframe(recurring_df, use_container_width=True, hide_index=True)
    st.caption("Sensors sorted by number of threshold crossing events. Higher crossing counts indicate recurring thermal issues.")

    thresh_takeaway_parts = []
    total_breaches = breach_df["Breach Count"].sum()
    if total_breaches > 0:
        worst_breach = breach_df.loc[breach_df["Breach Count"].idxmax()]
        thresh_takeaway_parts.append(f"**{int(total_breaches)}** total threshold breach readings detected.")
        thresh_takeaway_parts.append(f"**{worst_breach['Sensor']}** has the most breaches (**{int(worst_breach['Breach Count'])}** readings above its {worst_breach['Threshold (°C)']}°C threshold).")
        max_margin = breach_df.loc[breach_df["Margin (°C)"].idxmax()]
        if max_margin["Margin (°C)"] > 0:
            thresh_takeaway_parts.append(f"Largest threshold exceedance: **{max_margin['Sensor']}** peaked **{max_margin['Margin (°C)']:.1f}°C** above its limit.")
    else:
        thresh_takeaway_parts.append("No threshold breaches detected — all sensors remained within their per-sensor threshold limits.")
    st.info("📌 **Key Takeaway:** " + " ".join(thresh_takeaway_parts))

    st.divider()

    st.header("Correlation Analysis")
    st.markdown("_Cross-sensor temperature correlation analysis to identify thermal relationships and co-occurring events._")
    with st.expander("ℹ️ What is correlation analysis?"):
        st.markdown("""
**What it shows:** How temperature readings from different sensors relate to each other over time.

**How to read it:** Correlation values range from -1 to +1. Values close to +1 mean sensors heat up and cool down together. Values close to -1 mean inverse behavior. Values near 0 mean no relationship.

**Why it matters:** Highly correlated sensors may share thermal pathways. Unexpected correlations or lack thereof can reveal insulation issues, airflow patterns, or sensor placement effects.

**What is the Jaccard Index?** The Jaccard Index (shown in the Spike Co-occurrence table) measures how often two sensors spike at the same time relative to how often either one spikes. It ranges from 0 to 1 — a value of 0 means the sensors never spike together, while 1 means every spike event is shared. For example, a Jaccard Index of 0.4 means 40% of all spike events (from either sensor) occurred simultaneously. High values suggest the sensors share a common thermal influence or are physically close to the same heat source.
""")

    if len(selected_sensors) >= 2:
        corr_pivot = filtered_df.pivot_table(values="Temperature", index="time", columns="Sensor Id", aggfunc="mean")
        corr_pivot.columns = [get_label(c) for c in corr_pivot.columns]
        corr_matrix = corr_pivot.corr()

        corr_col1, corr_col2 = st.columns(2)
        with corr_col1:
            st.subheader("Correlation Heatmap")
            fig_corr = px.imshow(corr_matrix.values,
                x=list(corr_matrix.columns), y=list(corr_matrix.index),
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                labels=dict(color="Correlation"),
                aspect="auto")
            fig_corr.update_layout(height=max(400, len(corr_matrix) * 40 + 150),
                template="plotly_white", margin=dict(l=80, r=30, t=40, b=80))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Red = strong positive correlation. Blue = strong negative correlation. White = no correlation.")

        with corr_col2:
            st.subheader("Same-Type vs Cross-Type Correlation")
            type_corr_data = []
            for i_idx, s1 in enumerate(selected_sensors):
                for j_idx, s2 in enumerate(selected_sensors):
                    if i_idx >= j_idx:
                        continue
                    l1, l2 = get_label(s1), get_label(s2)
                    t1, t2 = sensor_type_map.get(s1, "Other"), sensor_type_map.get(s2, "Other")
                    if l1 in corr_matrix.columns and l2 in corr_matrix.columns:
                        c = corr_matrix.loc[l1, l2]
                        comp_type = "Same Type" if t1 == t2 else "Cross Type"
                        type_corr_data.append({
                            "Pair": f"{l1} — {l2}",
                            "Correlation": round(c, 3),
                            "Type 1": t1, "Type 2": t2,
                            "Comparison": comp_type,
                        })
            if type_corr_data:
                type_corr_df = pd.DataFrame(type_corr_data)
                fig_type_corr = px.bar(type_corr_df, x="Pair", y="Correlation", color="Comparison",
                    color_discrete_map={"Same Type": "#3498DB", "Cross Type": "#E8574A"},
                    labels={"Pair": "Sensor Pair", "Correlation": "Correlation Coefficient"})
                fig_type_corr.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
                fig_type_corr.update_layout(height=400, template="plotly_white",
                    xaxis=dict(tickangle=-45), margin=dict(l=40, r=20, t=40, b=100))
                st.plotly_chart(fig_type_corr, use_container_width=True)
                st.caption("Blue bars = same sensor type pairs. Red bars = cross-type pairs.")
            else:
                st.info("Not enough sensor pairs for comparison.")

        tm_sensors = [s for s in selected_sensors if sensor_type_map.get(s) == "TM"]
        non_tm_sensors = [s for s in selected_sensors if sensor_type_map.get(s) in ("MSU", "Axle")]
        if tm_sensors and non_tm_sensors:
            st.subheader("TM Sensors vs Nearby MSU/Axle Sensors")
            tm_comparison = []
            for tm_s in tm_sensors:
                tm_label = get_label(tm_s)
                for other_s in non_tm_sensors:
                    other_label = get_label(other_s)
                    if tm_label in corr_matrix.columns and other_label in corr_matrix.columns:
                        c = corr_matrix.loc[tm_label, other_label]
                        tm_comparison.append({
                            "TM Sensor": tm_label,
                            "Compared With": other_label,
                            "Type": sensor_type_map.get(other_s, "Other"),
                            "Correlation": round(c, 3),
                        })
            if tm_comparison:
                tm_comp_df = pd.DataFrame(tm_comparison)
                fig_tm = px.bar(tm_comp_df, x="Compared With", y="Correlation", color="TM Sensor",
                    barmode="group", labels={"Compared With": "MSU/Axle Sensor", "Correlation": "Correlation"},
                    color_discrete_sequence=px.colors.qualitative.Set2)
                fig_tm.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig_tm, use_container_width=True)
                st.caption("How each TM sensor correlates with nearby MSU and Axle sensors.")

        st.subheader("Spike Co-occurrence Analysis")
        spike_data_corr = filtered_df.pivot_table(values="Is Spike", index="time", columns="Sensor Id", aggfunc="max").fillna(False).astype(int)
        spike_data_corr.columns = [get_label(c) for c in spike_data_corr.columns]
        if spike_data_corr.shape[1] >= 2:
            co_occur = []
            cols_list = list(spike_data_corr.columns)
            for i_idx in range(len(cols_list)):
                for j_idx in range(i_idx + 1, len(cols_list)):
                    s1_spikes = spike_data_corr[cols_list[i_idx]]
                    s2_spikes = spike_data_corr[cols_list[j_idx]]
                    both = int((s1_spikes & s2_spikes).sum())
                    either = int((s1_spikes | s2_spikes).sum())
                    jaccard = round(both / either, 3) if either > 0 else 0
                    co_occur.append({
                        "Sensor A": cols_list[i_idx],
                        "Sensor B": cols_list[j_idx],
                        "Co-occurring Spikes": both,
                        "Jaccard Index": jaccard,
                    })
            co_occur_df = pd.DataFrame(co_occur).sort_values("Co-occurring Spikes", ascending=False)
            if co_occur_df["Co-occurring Spikes"].sum() > 0:
                fig_cooccur = px.bar(co_occur_df.head(15), x="Sensor A", y="Co-occurring Spikes",
                    color="Sensor B", barmode="group",
                    labels={"Co-occurring Spikes": "Simultaneous Spikes"},
                    color_discrete_sequence=px.colors.qualitative.Set2)
                fig_cooccur.update_layout(height=400, template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig_cooccur, use_container_width=True)
                st.caption("Sensor pairs that experience spikes at the same time. High co-occurrence suggests shared thermal influences.")
            else:
                st.info("No co-occurring spike events detected.")
            st.dataframe(co_occur_df, use_container_width=True, hide_index=True)
        else:
            st.info("Need at least 2 sensors for spike co-occurrence analysis.")

        corr_takeaway_parts = []
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
        avg_corr = upper_tri.stack().mean()
        corr_takeaway_parts.append(f"Average cross-sensor correlation: **{avg_corr:.3f}**.")
        max_corr_val = upper_tri.stack().max()
        max_corr_pair = upper_tri.stack().idxmax()
        corr_takeaway_parts.append(f"Strongest correlation: **{max_corr_pair[0]}** & **{max_corr_pair[1]}** (r = {max_corr_val:.3f}).")
        min_corr_val = upper_tri.stack().min()
        min_corr_pair = upper_tri.stack().idxmin()
        corr_takeaway_parts.append(f"Weakest correlation: **{min_corr_pair[0]}** & **{min_corr_pair[1]}** (r = {min_corr_val:.3f}).")
        if avg_corr > 0.7:
            corr_takeaway_parts.append("High overall correlation suggests sensors share common thermal environments or load patterns.")
        elif avg_corr < 0.3:
            corr_takeaway_parts.append("Low overall correlation indicates sensors operate largely independently — different thermal zones or load profiles.")
        st.info("📌 **Key Takeaway:** " + " ".join(corr_takeaway_parts))
    else:
        st.info("Select at least 2 sensors for correlation analysis.")


st.header("Export PDF Report")
st.markdown("_Generate a professional A0-landscape PDF report suitable for poster printing at customer meetings. The report includes one page per sensor plus comparative analysis._")
with st.expander("ℹ️ About the PDF report"):
    st.markdown("""
**What you get:** A multi-page professional report in A0 landscape format designed for large-format printing.

**Report structure:**
1. **Executive Summary** — KPIs, key insights, and operational observations
2. **Combined Sensor Overview** — All sensors on one chart
3. **Individual Sensor Pages** — One full page per sensor with detailed analysis
4. **Comparative Analysis** — Rankings and fleet-wide comparison
5. **Alert Summary** — Per-sensor alerts, daily and monthly alert trends
6. **Temperature Heatmap** — Sensor × time thermal mapping
7. **Spike Analysis** — Rate of change analysis and spike detection
8. **Temperature Distribution** — Histogram and box plot analysis
9. **Temporal Patterns** — Hourly, daily, and monthly temperature patterns
10. **Threshold & Correlation Analysis** — Per-sensor thresholds and cross-sensor correlation (when Legend_Master is provided)

**Print recommendations:** Use A0 landscape format for best results. The report is designed as a poster-style document for client presentations.
""")


def generate_pdf_report(filtered_df, alerts_df, sensor_stats, fig_temp_combined,
                        fig_hist, fig_box, fig_roc, fig_heatmap,
                        selected_sensors, total_alerts, total_records,
                        alert_pct, filtered_min_ts, filtered_max_ts,
                        spike_count, insights_list,
                        has_legend=False, sensor_legend_map=None, sensor_threshold_map=None,
                        sensor_type_map=None, sensor_description_map=None, legend_df=None):
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                     Table, TableStyle, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    A0_LANDSCAPE = (1189*mm, 841*mm)
    page_w, page_h = A0_LANDSCAPE

    PRIMARY = "#1B2A4A"
    ACCENT = "#E8574A"
    BLUE = "#3498DB"
    DARK_TEXT = "#2C3E50"
    LIGHT_TEXT = "#7F8C8D"
    BG_LIGHT = "#F5F7FA"
    BG_CARD = "#FFFFFF"
    BORDER = "#DDE1E7"
    GREEN = "#27AE60"

    buf = io.BytesIO()
    margin_lr = 2.5*cm
    margin_tb = 2*cm
    doc = SimpleDocTemplate(buf, pagesize=A0_LANDSCAPE,
                            leftMargin=margin_lr, rightMargin=margin_lr,
                            topMargin=margin_tb + 1.8*cm,
                            bottomMargin=margin_tb + 1*cm)

    page_num_holder = [0]
    report_title_text = "Railway Locomotive Temperature Sensor Analysis"
    period_str = f"{str(filtered_min_ts)[:10]}  to  {str(filtered_max_ts)[:10]}"

    def draw_page_header_footer(canvas_obj, doc_obj):
        page_num_holder[0] += 1
        canvas_obj.saveState()
        canvas_obj.setFillColor(HexColor(PRIMARY))
        canvas_obj.rect(0, page_h - 1.6*cm, page_w, 1.6*cm, fill=1, stroke=0)
        canvas_obj.setFillColor(white)
        canvas_obj.setFont("Helvetica-Bold", 28)
        canvas_obj.drawString(margin_lr, page_h - 1.1*cm, report_title_text)
        canvas_obj.setFont("Helvetica", 22)
        canvas_obj.drawRightString(page_w - margin_lr, page_h - 1.1*cm, period_str)
        canvas_obj.setFillColor(HexColor(BORDER))
        canvas_obj.rect(0, 0, page_w, 0.8*cm, fill=1, stroke=0)
        canvas_obj.setFillColor(HexColor(DARK_TEXT))
        canvas_obj.setFont("Helvetica", 20)
        canvas_obj.drawString(margin_lr, 0.25*cm, "Confidential — For Internal Use Only")
        canvas_obj.drawRightString(page_w - margin_lr, 0.25*cm,
                                    f"Page {page_num_holder[0]}")
        canvas_obj.restoreState()

    styles = getSampleStyleSheet()

    cover_title = ParagraphStyle("CoverTitle", parent=styles["Title"],
        fontSize=120, leading=140, alignment=TA_CENTER,
        textColor=HexColor(PRIMARY), spaceAfter=30,
        fontName="Helvetica-Bold")
    cover_subtitle = ParagraphStyle("CoverSub", parent=styles["Normal"],
        fontSize=52, leading=64, alignment=TA_CENTER,
        textColor=HexColor(LIGHT_TEXT), spaceAfter=60)
    page_title = ParagraphStyle("PageTitle", parent=styles["Heading1"],
        fontSize=72, leading=88, alignment=TA_LEFT,
        textColor=HexColor(PRIMARY), spaceAfter=20, spaceBefore=0,
        fontName="Helvetica-Bold")
    page_desc = ParagraphStyle("PageDesc", parent=styles["Normal"],
        fontSize=32, leading=46, textColor=HexColor(DARK_TEXT), spaceAfter=24)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=30, leading=44, textColor=HexColor(DARK_TEXT), spaceAfter=16)
    body_large = ParagraphStyle("BodyLarge", parent=styles["Normal"],
        fontSize=36, leading=50, textColor=HexColor(DARK_TEXT), spaceAfter=20)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"],
        fontSize=26, leading=38, textColor=HexColor(LIGHT_TEXT),
        spaceAfter=10, alignment=TA_CENTER)
    insight_bullet = ParagraphStyle("InsightBullet", parent=styles["Normal"],
        fontSize=34, leading=50, textColor=HexColor(DARK_TEXT),
        spaceAfter=12, leftIndent=30, bulletIndent=0)
    kpi_value = ParagraphStyle("KpiValue", parent=styles["Normal"],
        fontSize=72, leading=86, alignment=TA_CENTER,
        textColor=HexColor(PRIMARY), fontName="Helvetica-Bold")
    kpi_label = ParagraphStyle("KpiLabel", parent=styles["Normal"],
        fontSize=28, leading=38, alignment=TA_CENTER,
        textColor=HexColor(LIGHT_TEXT))
    sensor_page_title = ParagraphStyle("SensorPageTitle", parent=styles["Heading1"],
        fontSize=84, leading=100, alignment=TA_LEFT,
        textColor=HexColor(PRIMARY), spaceAfter=20, fontName="Helvetica-Bold")
    metric_val = ParagraphStyle("MetricVal", parent=styles["Normal"],
        fontSize=60, leading=72, alignment=TA_CENTER,
        textColor=HexColor(PRIMARY), fontName="Helvetica-Bold")
    metric_lbl = ParagraphStyle("MetricLbl", parent=styles["Normal"],
        fontSize=26, leading=36, alignment=TA_CENTER,
        textColor=HexColor(LIGHT_TEXT))
    takeaway_title = ParagraphStyle("TakeawayTitle", parent=styles["Normal"],
        fontSize=36, leading=48, textColor=HexColor(PRIMARY),
        fontName="Helvetica-Bold", spaceAfter=8)
    takeaway_body = ParagraphStyle("TakeawayBody", parent=styles["Normal"],
        fontSize=30, leading=44, textColor=HexColor(DARK_TEXT))
    table_hdr = ParagraphStyle("TblHdr", parent=styles["Normal"],
        fontSize=30, leading=40, alignment=TA_CENTER,
        textColor=white, fontName="Helvetica-Bold")
    table_cell = ParagraphStyle("TblCell", parent=styles["Normal"],
        fontSize=28, leading=38, alignment=TA_CENTER,
        textColor=HexColor(DARK_TEXT))
    table_cell_l = ParagraphStyle("TblCellL", parent=table_cell, alignment=TA_LEFT)
    chart_note = ParagraphStyle("ChartNote", parent=styles["Normal"],
        fontSize=28, leading=40, textColor=HexColor(DARK_TEXT),
        spaceAfter=10, fontName="Helvetica-Oblique")

    W = page_w - 2 * margin_lr
    HERO_H = 1500
    MAIN_H = 1200
    SIDE_H = 900
    HALF_W = W * 0.48
    GAP = W * 0.04

    def pdf_label(sensor_id):
        if sensor_legend_map:
            return sensor_legend_map.get(sensor_id, str(sensor_id))
        return str(sensor_id)

    POSTER_FONT = dict(size=36, family="Arial, Helvetica, sans-serif")
    POSTER_TITLE = dict(size=44, family="Arial, Helvetica, sans-serif", color=DARK_TEXT)
    POSTER_AXIS = dict(size=32, family="Arial, Helvetica, sans-serif")
    POSTER_TICK = dict(size=28, family="Arial, Helvetica, sans-serif")
    POSTER_LEGEND = dict(size=30, family="Arial, Helvetica, sans-serif")
    POSTER_ANNOTATION = dict(size=30, family="Arial, Helvetica, sans-serif")
    POSTER_COLORBAR = dict(size=26, family="Arial, Helvetica, sans-serif")

    def style_fig(fig):
        fig.update_layout(
            font=POSTER_FONT,
            title_font=POSTER_TITLE,
            xaxis_title_font=POSTER_AXIS,
            yaxis_title_font=POSTER_AXIS,
            xaxis_tickfont=POSTER_TICK,
            yaxis_tickfont=POSTER_TICK,
            legend_font=POSTER_LEGEND,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#ECECEC", gridwidth=1, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#ECECEC", gridwidth=1, zeroline=False),
        )
        for ann in fig.layout.annotations or []:
            ann.font = POSTER_ANNOTATION
        try:
            if hasattr(fig.layout, 'coloraxis') and fig.layout.coloraxis.colorbar:
                fig.layout.coloraxis.colorbar.tickfont = POSTER_COLORBAR
                fig.layout.coloraxis.colorbar.title = dict(font=POSTER_COLORBAR)
        except Exception:
            pass

    def render(fig, w_px=3200, h_px=1800):
        img_bytes = fig.to_image(format="png", width=w_px, height=h_px, scale=2)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(img_bytes)
        tmp.close()
        return tmp.name

    temp_files = []
    story = []

    def make_kpi_card(value_text, label_text, accent=False):
        bg = HexColor(PRIMARY) if accent else HexColor(BG_CARD)
        val_color = white if accent else HexColor(PRIMARY)
        lbl_color = HexColor("#B0BEC5") if accent else HexColor(LIGHT_TEXT)
        v_style = ParagraphStyle("kv", parent=kpi_value, textColor=val_color)
        l_style = ParagraphStyle("kl", parent=kpi_label, textColor=lbl_color)
        data = [[Paragraph(value_text, v_style)], [Paragraph(label_text, l_style)]]
        t = Table(data, colWidths=[W/7 - 10])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("BOX", (0, 0), (-1, -1), 2, HexColor(BORDER) if not accent else HexColor(PRIMARY)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 28),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 28),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return t

    def make_takeaway_box(text):
        data = [
            [Paragraph("KEY TAKEAWAY", takeaway_title)],
            [Paragraph(text, takeaway_body)]
        ]
        t = Table(data, colWidths=[W])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F0F4FF")),
            ("BOX", (0, 0), (-1, -1), 2, HexColor(BLUE)),
            ("LINEABOVE", (0, 0), (-1, 0), 5, HexColor(BLUE)),
            ("TOPPADDING", (0, 0), (-1, -1), 20),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
            ("LEFTPADDING", (0, 0), (-1, -1), 30),
            ("RIGHTPADDING", (0, 0), (-1, -1), 30),
        ]))
        return t

    # ==================== PAGE 1: COVER ====================
    story.append(Spacer(1, 200))
    story.append(Paragraph("Railway Locomotive", cover_title))
    story.append(Paragraph("Temperature Sensor Analysis", ParagraphStyle(
        "CoverTitle2", parent=cover_title, fontSize=100, leading=120, spaceAfter=40)))
    story.append(Spacer(1, 40))
    story.append(Paragraph("Comprehensive Sensor Monitoring Report", cover_subtitle))
    story.append(Paragraph(f"Monitoring Period: {period_str}", ParagraphStyle(
        "CoverPeriod", parent=cover_subtitle, fontSize=40, textColor=HexColor(DARK_TEXT))))
    story.append(Spacer(1, 80))

    cards = [
        make_kpi_card(str(len(selected_sensors)), "Sensors"),
        make_kpi_card(f"{total_records:,}", "Readings"),
        make_kpi_card(str(total_alerts), "Alerts", accent=total_alerts > 0),
        make_kpi_card(str(spike_count), "Spikes", accent=spike_count > 10),
        make_kpi_card(f"{alert_pct:.1f}%", "Alert Rate"),
        make_kpi_card(f"{THRESHOLD:.0f}°C", "Threshold"),
        make_kpi_card(f"{sensor_stats['Max Temp (°C)'].max():.0f}°C", "Peak Temp", accent=True),
    ]
    kpi_row = Table([cards], colWidths=[W/7]*7)
    kpi_row.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(kpi_row)
    story.append(Spacer(1, 60))

    story.append(Paragraph("Key Findings", ParagraphStyle(
        "KFTitle", parent=page_title, fontSize=56, spaceAfter=16)))
    for ins in insights_list:
        clean = ins.replace("✅", "").replace("⚠️", "").replace("📈", "").replace("🌡️", "").replace("⚡", "").replace("🕐", "")
        clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', clean)
        story.append(Paragraph(clean.strip(), insight_bullet))
    story.append(Spacer(1, 40))

    explain_data = [
        [Paragraph("ALERTS (Threshold Crossings)", ParagraphStyle("eh", parent=table_hdr, fontSize=32)),
         Paragraph("SPIKES (Rapid Changes)", ParagraphStyle("eh2", parent=table_hdr, fontSize=32))],
        [Paragraph("Triggered when temperature rises above 80°C. Uses crossing-event logic — counted once per excursion above threshold.", body_style),
         Paragraph("Triggered when temperature changes more than ±5°C between consecutive readings. Early warning signal of instability.", body_style)],
    ]
    explain_tbl = Table(explain_data, colWidths=[W/2]*2)
    explain_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(PRIMARY)),
        ("BACKGROUND", (0, 1), (-1, 1), HexColor(BG_LIGHT)),
        ("BOX", (0, 0), (-1, -1), 2, HexColor(BORDER)),
        ("INNERGRID", (0, 0), (-1, -1), 1, HexColor(BORDER)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
    ]))
    story.append(explain_tbl)
    story.append(PageBreak())

    # ==================== PAGE 2: COMBINED OVERVIEW ====================
    story.append(Paragraph("Combined Sensor Overview", page_title))
    story.append(Paragraph(
        f"All {len(selected_sensors)} sensors with rolling averages. "
        f"Zone shading: Safe &lt; {SAFE_LIMIT}°C | Warning {SAFE_LIMIT}–{WARNING_LIMIT}°C | Critical &gt; {WARNING_LIMIT}°C",
        page_desc))

    fig_temp_combined.update_traces(selector=dict(mode="lines"), line=dict(width=4))
    fig_temp_combined.update_layout(
        xaxis=dict(nticks=12), margin=dict(l=120, r=80, t=60, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=POSTER_LEGEND, bgcolor="rgba(255,255,255,0.8)"),
    )
    style_fig(fig_temp_combined)
    img = render(fig_temp_combined, 3400, 1900)
    temp_files.append(img)
    story.append(Image(img, width=W, height=HERO_H))
    story.append(Paragraph(
        "Temperature readings with zone shading. Dotted lines = rolling averages. Red dashed = threshold.",
        caption_style))
    story.append(PageBreak())

    # ==================== PER-SENSOR PAGES ====================
    overall_avg = sensor_stats["Avg Temp (°C)"].mean()
    overall_max = sensor_stats["Max Temp (°C)"].max()
    overall_spike_avg = sensor_stats["Spike Events"].mean()

    for sensor_id in selected_sensors:
        s_data = filtered_df[filtered_df["Sensor Id"] == sensor_id].copy()
        s_stats = sensor_stats[sensor_stats["Sensor Id"] == sensor_id].iloc[0]
        s_alerts = int(s_stats["Alert Events"])
        s_spikes = int(s_stats["Spike Events"])
        s_min = s_stats["Min Temp (°C)"]
        s_max = s_stats["Max Temp (°C)"]
        s_avg = s_stats["Avg Temp (°C)"]

        story.append(Paragraph(f"{pdf_label(sensor_id)}", sensor_page_title))

        m_cards = [
            [Paragraph(f"{s_min:.1f}°C", metric_val), Paragraph(f"{s_max:.1f}°C", metric_val),
             Paragraph(f"{s_avg:.1f}°C", metric_val), Paragraph(str(s_alerts), metric_val),
             Paragraph(str(s_spikes), metric_val), Paragraph(f"{len(s_data):,}", metric_val)],
            [Paragraph("Min Temp", metric_lbl), Paragraph("Max Temp", metric_lbl),
             Paragraph("Avg Temp", metric_lbl), Paragraph("Alert Events", metric_lbl),
             Paragraph("Spike Events", metric_lbl), Paragraph("Readings", metric_lbl)],
        ]
        m_tbl = Table(m_cards, colWidths=[W/6]*6)
        m_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor(BG_LIGHT)),
            ("BOX", (0, 0), (-1, -1), 2, HexColor(BORDER)),
            ("INNERGRID", (0, 0), (-1, -1), 1, HexColor(BORDER)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, 0), 18),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 1), (-1, 1), 2),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 14),
        ]))
        story.append(m_tbl)
        story.append(Spacer(1, 10))

        y_max_s = max(s_data["Temperature"].max(), 120)
        fig_s = go.Figure()
        fig_s.add_hrect(y0=0, y1=SAFE_LIMIT, fillcolor="green", opacity=0.06, line_width=0)
        fig_s.add_hrect(y0=SAFE_LIMIT, y1=WARNING_LIMIT, fillcolor="orange", opacity=0.06, line_width=0)
        fig_s.add_hrect(y0=WARNING_LIMIT, y1=y_max_s*1.1, fillcolor="red", opacity=0.06, line_width=0)
        fig_s.add_trace(go.Scatter(x=s_data["time"], y=s_data["Temperature"],
            mode="lines", name="Temperature", line=dict(color=BLUE, width=4)))
        fig_s.add_trace(go.Scatter(x=s_data["time"], y=s_data["Rolling Avg"],
            mode="lines", name="Rolling Avg", line=dict(color=ACCENT, width=4, dash="dot")))
        fig_s.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", line_width=3,
            annotation_text=f"Threshold ({THRESHOLD}°C)")
        fig_s.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)",
            template="plotly_white", margin=dict(l=120, r=80, t=50, b=100),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(nticks=10))
        style_fig(fig_s)
        img = render(fig_s, 3400, 1400)
        temp_files.append(img)
        story.append(Image(img, width=W, height=850))
        story.append(Spacer(1, 10))

        fig_hist_s = px.histogram(s_data, x="Temperature", nbins=40, color_discrete_sequence=[BLUE])
        fig_hist_s.add_vline(x=THRESHOLD, line_dash="dash", line_color="red", line_width=3)
        fig_hist_s.update_layout(template="plotly_white", title="Temperature Distribution",
            margin=dict(l=100, r=60, t=90, b=80))
        style_fig(fig_hist_s)

        fig_roc_s = go.Figure()
        s_normal = s_data[~s_data["Is Spike"]]
        s_spikes_d = s_data[s_data["Is Spike"]]
        fig_roc_s.add_trace(go.Scatter(x=s_normal["time"], y=s_normal["Temp Change"],
            mode="markers", name="Normal", marker=dict(color=BLUE, size=8, opacity=0.5)))
        if not s_spikes_d.empty:
            fig_roc_s.add_trace(go.Scatter(x=s_spikes_d["time"], y=s_spikes_d["Temp Change"],
                mode="markers", name="Spikes", marker=dict(color=ACCENT, size=18, symbol="diamond", opacity=0.9)))
        fig_roc_s.add_hline(y=SPIKE_THRESHOLD_RATE, line_dash="dot", line_color="red", line_width=3)
        fig_roc_s.add_hline(y=-SPIKE_THRESHOLD_RATE, line_dash="dot", line_color="red", line_width=3)
        fig_roc_s.update_layout(xaxis_title="Time", yaxis_title="Change (°C)",
            template="plotly_white", title="Rate of Change & Spikes",
            margin=dict(l=100, r=60, t=90, b=80), xaxis=dict(nticks=8))
        style_fig(fig_roc_s)

        img_h = render(fig_hist_s, 1600, 1100)
        img_r = render(fig_roc_s, 1600, 1100)
        temp_files.extend([img_h, img_r])
        pair = Table(
            [[Image(img_h, width=HALF_W, height=650), Image(img_r, width=HALF_W, height=650)]],
            colWidths=[HALF_W + GAP/2, HALF_W + GAP/2])
        pair.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
        story.append(pair)
        story.append(Spacer(1, 8))

        obs = []
        slbl = pdf_label(sensor_id)
        td = s_avg - overall_avg
        if abs(td) < 1.0:
            obs.append(f"{slbl} operates near the fleet average ({overall_avg:.1f}°C).")
        elif td > 0:
            obs.append(f"{slbl} runs <b>{td:.1f}°C hotter</b> than fleet average ({overall_avg:.1f}°C).")
        else:
            obs.append(f"{slbl} runs <b>{abs(td):.1f}°C cooler</b> than fleet average ({overall_avg:.1f}°C).")
        if s_max == overall_max:
            obs.append(f"Recorded the <b>highest peak</b> across all sensors ({s_max:.1f}°C).")
        elif s_max > THRESHOLD:
            obs.append(f"Peak ({s_max:.1f}°C) <b>exceeded threshold</b>.")
        else:
            obs.append(f"Peak ({s_max:.1f}°C) within safe limits.")
        if s_spikes > overall_spike_avg * 1.5:
            obs.append(f"{s_spikes} spikes — <b>more volatile than average</b> ({overall_spike_avg:.0f} avg).")
        elif s_spikes == 0:
            obs.append("<b>Very stable</b> — no spike events.")
        else:
            obs.append(f"{s_spikes} spike events (fleet avg: {overall_spike_avg:.0f}).")
        story.append(make_takeaway_box(" ".join(obs)))
        story.append(PageBreak())

    # ==================== COMPARATIVE ANALYSIS ====================
    story.append(Paragraph("Comparative Analysis", page_title))
    story.append(Paragraph("Sensor performance rankings and fleet-wide comparison.", page_desc))

    if has_legend:
        tbl_data = [[Paragraph("Sensor", table_hdr), Paragraph("Type", table_hdr),
                      Paragraph("Min (°C)", table_hdr), Paragraph("Max (°C)", table_hdr),
                      Paragraph("Avg (°C)", table_hdr), Paragraph("Threshold", table_hdr),
                      Paragraph("Alerts", table_hdr), Paragraph("Spikes", table_hdr)]]
        for _, row in sensor_stats.iterrows():
            tbl_data.append([
                Paragraph(pdf_label(row["Sensor Id"]), table_cell_l),
                Paragraph(str(row.get("Sensor Type", "")), table_cell),
                Paragraph(f"{row['Min Temp (°C)']:.1f}", table_cell),
                Paragraph(f"{row['Max Temp (°C)']:.1f}", table_cell),
                Paragraph(f"{row['Avg Temp (°C)']:.1f}", table_cell),
                Paragraph(f"{row.get('Sensor Threshold', THRESHOLD):.0f}°C", table_cell),
                Paragraph(str(row["Alert Events"]), table_cell),
                Paragraph(str(row["Spike Events"]), table_cell),
            ])
        cw = [W*0.16, W*0.10, W*0.10, W*0.10, W*0.10, W*0.12, W*0.12, W*0.12]
    else:
        tbl_data = [[Paragraph("Sensor Id", table_hdr), Paragraph("Min (°C)", table_hdr),
                      Paragraph("Max (°C)", table_hdr), Paragraph("Avg (°C)", table_hdr),
                      Paragraph("Alerts", table_hdr), Paragraph("Spikes", table_hdr)]]
        for _, row in sensor_stats.iterrows():
            tbl_data.append([
                Paragraph(str(row["Sensor Id"]), table_cell_l),
                Paragraph(f"{row['Min Temp (°C)']:.1f}", table_cell),
                Paragraph(f"{row['Max Temp (°C)']:.1f}", table_cell),
                Paragraph(f"{row['Avg Temp (°C)']:.1f}", table_cell),
                Paragraph(str(row["Alert Events"]), table_cell),
                Paragraph(str(row["Spike Events"]), table_cell),
            ])
        cw = [W*0.22, W*0.15, W*0.15, W*0.15, W*0.15, W*0.15]
    stbl = Table(tbl_data, colWidths=cw)
    stbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(PRIMARY)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor(BG_CARD), HexColor(BG_LIGHT)]),
        ("BOX", (0, 0), (-1, -1), 2, HexColor(BORDER)),
        ("INNERGRID", (0, 0), (-1, -1), 1, HexColor(BORDER)),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(stbl)
    story.append(Spacer(1, 30))

    rk_df = sensor_stats.copy()
    rk_df["Label"] = rk_df["Sensor Id"].map(lambda x: pdf_label(x))

    fig_rk1 = px.bar(rk_df.sort_values("Max Temp (°C)", ascending=True),
        x="Max Temp (°C)", y="Label", orientation="h",
        color="Max Temp (°C)", color_continuous_scale="RdYlGn_r",
        title="Peak Temperature by Sensor")
    fig_rk1.add_vline(x=THRESHOLD, line_dash="dash", line_color="red", line_width=3,
        annotation_text=f"Threshold ({THRESHOLD}°C)")
    fig_rk1.update_layout(template="plotly_white", yaxis_type="category", showlegend=False,
        margin=dict(l=140, r=60, t=90, b=80))
    style_fig(fig_rk1)

    fig_rk2 = px.bar(rk_df.sort_values("Spike Events", ascending=True),
        x="Spike Events", y="Label", orientation="h",
        color="Spike Events", color_continuous_scale="YlOrRd",
        title="Spike Events by Sensor")
    fig_rk2.update_layout(template="plotly_white", yaxis_type="category", showlegend=False,
        margin=dict(l=140, r=60, t=90, b=80))
    style_fig(fig_rk2)

    img1 = render(fig_rk1, 1600, 1300)
    img2 = render(fig_rk2, 1600, 1300)
    temp_files.extend([img1, img2])
    rk_pair = Table(
        [[Image(img1, width=HALF_W, height=SIDE_H+100), Image(img2, width=HALF_W, height=SIDE_H+100)]],
        colWidths=[HALF_W + GAP/2, HALF_W + GAP/2])
    rk_pair.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(rk_pair)
    story.append(Paragraph(
        "Left: peak temperature ranking (red line = threshold). Right: spike event ranking.",
        caption_style))

    hottest_sensor = sensor_stats.loc[sensor_stats["Max Temp (°C)"].idxmax()]
    most_spiky = sensor_stats.loc[sensor_stats["Spike Events"].idxmax()]
    coolest_sensor = sensor_stats.loc[sensor_stats["Avg Temp (°C)"].idxmin()]
    comp_parts = []
    comp_parts.append(f"<b>{pdf_label(hottest_sensor['Sensor Id'])}</b> recorded the highest peak at <b>{hottest_sensor['Max Temp (°C)']:.1f}°C</b>.")
    comp_parts.append(f"<b>{pdf_label(most_spiky['Sensor Id'])}</b> is the most volatile with <b>{int(most_spiky['Spike Events'])} spike events</b>.")
    comp_parts.append(f"<b>{pdf_label(coolest_sensor['Sensor Id'])}</b> has the lowest average temperature ({coolest_sensor['Avg Temp (°C)']:.1f}°C) — the most stable performer.")
    above_thresh = sensor_stats[sensor_stats["Max Temp (°C)"] > THRESHOLD]
    if len(above_thresh) > 0:
        comp_parts.append(f"<b>{len(above_thresh)} of {len(sensor_stats)} sensors</b> exceeded the {THRESHOLD}°C threshold.")
    else:
        comp_parts.append(f"No sensors exceeded the {THRESHOLD}°C threshold — all within safe limits.")
    story.append(make_takeaway_box(" ".join(comp_parts)))
    story.append(PageBreak())

    # ==================== ALERT SUMMARY ====================
    story.append(Paragraph("Alert Summary", page_title))
    story.append(Paragraph(
        f"Threshold crossing event analysis. {total_alerts} alert events detected across "
        f"{total_records:,} readings ({alert_pct:.2f}% alert rate).", page_desc))

    if not alerts_df.empty:
        alerts_per_s = alerts_df.groupby("Sensor Id").size().reset_index(name="Alert Events")
        alerts_per_s["Label"] = alerts_per_s["Sensor Id"].map(
            lambda s: pdf_label(s) if has_legend else str(s))
        fig_alerts_pdf = px.bar(alerts_per_s, x="Label", y="Alert Events",
                                 color="Alert Events", color_continuous_scale="Reds",
                                 labels={"Label": "Sensor", "Alert Events": "Alert Events"})
        fig_alerts_pdf.update_layout(height=500, template="plotly_white",
                                      xaxis_type="category", margin=dict(l=80, r=40, t=40, b=80))
        style_fig(fig_alerts_pdf)
        img = render(fig_alerts_pdf, 1600, 1200)
        temp_files.append(img)

        alerts_df_copy = alerts_df.copy()
        alerts_df_copy["Alert Date"] = alerts_df_copy["Alert Time"].dt.date
        daily_a = alerts_df_copy.groupby("Alert Date").size().reset_index(name="Alert Events")
        fig_daily_pdf = px.bar(daily_a, x="Alert Date", y="Alert Events",
                                color_discrete_sequence=["#E45756"])
        fig_daily_pdf.update_layout(height=500, template="plotly_white",
                                     margin=dict(l=80, r=40, t=40, b=80))
        style_fig(fig_daily_pdf)
        img2 = render(fig_daily_pdf, 1600, 1200)
        temp_files.append(img2)

        alert_pair = Table(
            [[Image(img, width=HALF_W, height=SIDE_H),
              Image(img2, width=HALF_W, height=SIDE_H)]],
            colWidths=[HALF_W, HALF_W])
        alert_pair.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(alert_pair)
        story.append(Paragraph(
            "Left: Alert events per sensor. Right: Daily alert trend over time.", caption_style))
        story.append(Spacer(1, 20))

        alerts_df_copy["Alert Month"] = alerts_df_copy["Alert Time"].dt.to_period("M").astype(str)
        monthly_a = alerts_df_copy.groupby("Alert Month").size().reset_index(name="Alert Events")
        fig_monthly_pdf = px.bar(monthly_a, x="Alert Month", y="Alert Events",
                                  color_discrete_sequence=["#FF6B6B"])
        fig_monthly_pdf.update_layout(height=500, template="plotly_white",
                                       margin=dict(l=80, r=40, t=40, b=80))
        style_fig(fig_monthly_pdf)
        img3 = render(fig_monthly_pdf, 2400, 1200)
        temp_files.append(img3)
        story.append(Image(img3, width=W*0.75, height=SIDE_H))
        story.append(Paragraph("Monthly alert event trend.", caption_style))
    else:
        story.append(Paragraph(
            "No alert events detected — all sensors remained within safe operating limits "
            "throughout the monitoring period.", body_large))

    alert_tk_parts = []
    if total_alerts > 0:
        alert_s_counts = sensor_stats[["Display Label", "Alert Events"]].sort_values("Alert Events", ascending=False)
        top_a = alert_s_counts.iloc[0]
        sensors_with_alerts = int((alert_s_counts["Alert Events"] > 0).sum())
        alert_tk_parts.append(f"<b>{total_alerts}</b> alert events across <b>{sensors_with_alerts}</b> sensors.")
        alert_tk_parts.append(f"<b>{top_a['Display Label']}</b> has the most alerts ({int(top_a['Alert Events'])} crossing events).")
        if alert_pct > 5:
            alert_tk_parts.append("Alert rate exceeds 5% — systematic thermal issues require investigation.")
        elif alert_pct < 1:
            alert_tk_parts.append("Alert rate below 1% — threshold crossings are infrequent.")
    else:
        alert_tk_parts.append("No alert events detected — all sensors operated within safe limits.")
    story.append(make_takeaway_box(" ".join(alert_tk_parts)))
    story.append(PageBreak())

    # ==================== HEATMAP ====================
    story.append(Paragraph("Temperature Heatmap", page_title))
    story.append(Paragraph(
        "Average temperature across sensors and time periods. Red/orange = hot. Green = cool.",
        page_desc))

    if fig_heatmap is not None:
        fig_heatmap.update_layout(margin=dict(l=120, r=80, t=60, b=100), xaxis=dict(nticks=14))
        style_fig(fig_heatmap)
        img = render(fig_heatmap, 3400, 2000)
        temp_files.append(img)
        story.append(Image(img, width=W, height=HERO_H))
        story.append(Paragraph(
            "Horizontal bands = consistently hot sensors. Vertical bands = fleet-wide hot periods.",
            caption_style))
    else:
        story.append(Paragraph("Insufficient data to generate heatmap.", body_style))

    hm_parts = []
    sensor_avgs = filtered_df.groupby("Sensor Id")["Temperature"].mean()
    hottest_s = sensor_avgs.idxmax()
    coolest_s = sensor_avgs.idxmin()
    hm_parts.append(f"<b>Sensor {hottest_s}</b> runs hottest overall ({sensor_avgs[hottest_s]:.1f}°C avg), while <b>Sensor {coolest_s}</b> runs coolest ({sensor_avgs[coolest_s]:.1f}°C avg).")
    if "time" in filtered_df.columns:
        hourly_fleet = filtered_df.groupby(filtered_df["time"].dt.hour)["Temperature"].mean()
        peak_hr = hourly_fleet.idxmax()
        cool_hr = hourly_fleet.idxmin()
        hm_parts.append(f"Fleet-wide peak hour: <b>{peak_hr}:00</b> ({hourly_fleet[peak_hr]:.1f}°C). Coolest hour: <b>{cool_hr}:00</b> ({hourly_fleet[cool_hr]:.1f}°C).")
    spread = sensor_avgs.max() - sensor_avgs.min()
    if spread > 10:
        hm_parts.append(f"Temperature spread across sensors is <b>{spread:.1f}°C</b> — significant variation warrants investigation.")
    else:
        hm_parts.append(f"Temperature spread across sensors is <b>{spread:.1f}°C</b> — fleet is relatively uniform.")
    story.append(make_takeaway_box(" ".join(hm_parts)))
    story.append(PageBreak())

    # ==================== SPIKE ANALYSIS ====================
    story.append(Paragraph("Spike Analysis", page_title))
    story.append(Paragraph(
        f"Rate of change analysis: ±{SPIKE_THRESHOLD_RATE}°C threshold. "
        f"{spike_count} spike events detected across all sensors.", page_desc))

    fig_roc.update_traces(selector=dict(mode="markers"), marker=dict(size=9))
    fig_roc.update_traces(selector=dict(name="Spikes"), marker=dict(size=18))
    fig_roc.update_layout(margin=dict(l=120, r=80, t=60, b=100), xaxis=dict(nticks=12))
    style_fig(fig_roc)
    img = render(fig_roc, 3400, 1600)
    temp_files.append(img)
    story.append(Image(img, width=W, height=MAIN_H))
    story.append(Spacer(1, 20))

    spk_df = sensor_stats.copy()
    spk_df["Sensor Id"] = spk_df["Sensor Id"].astype(str)
    fig_spk = px.bar(spk_df, x="Sensor Id", y="Spike Events",
        color="Spike Events", color_continuous_scale="YlOrRd", title="Spike Events per Sensor")
    fig_spk.update_layout(template="plotly_white", xaxis_type="category",
        margin=dict(l=100, r=60, t=90, b=80))
    style_fig(fig_spk)
    img = render(fig_spk, 3000, 1200)
    temp_files.append(img)
    story.append(Image(img, width=W*0.85, height=SIDE_H))
    story.append(Paragraph(
        "Sensors with more spike events may need closer monitoring or preventive maintenance.",
        caption_style))

    sp_parts = []
    sp_parts.append(f"<b>{spike_count} spike events</b> detected across all sensors (rate-of-change > ±{SPIKE_THRESHOLD_RATE}°C).")
    if spike_count > 0:
        spk_max_sensor = sensor_stats.loc[sensor_stats["Spike Events"].idxmax()]
        sp_parts.append(f"<b>Sensor {spk_max_sensor['Sensor Id']}</b> leads with <b>{int(spk_max_sensor['Spike Events'])} spikes</b> — consider prioritised inspection.")
        zero_spike = sensor_stats[sensor_stats["Spike Events"] == 0]
        if len(zero_spike) > 0:
            sp_parts.append(f"{len(zero_spike)} sensor(s) recorded <b>zero spikes</b> — very stable operation.")
        avg_spk = sensor_stats["Spike Events"].mean()
        sp_parts.append(f"Fleet average: <b>{avg_spk:.1f} spikes per sensor</b>.")
    else:
        sp_parts.append("No rapid temperature changes detected — all sensors show <b>smooth thermal behaviour</b>.")
    story.append(make_takeaway_box(" ".join(sp_parts)))
    story.append(PageBreak())

    # ==================== DISTRIBUTION ====================
    story.append(Paragraph("Temperature Distribution", page_title))
    story.append(Paragraph("Histogram and box plot analysis of temperature readings.", page_desc))

    fig_hist.update_layout(margin=dict(l=120, r=80, t=90, b=100), title="Temperature Histogram")
    style_fig(fig_hist)
    img_h = render(fig_hist, 1600, 1300)
    temp_files.append(img_h)

    fig_box.update_layout(margin=dict(l=120, r=80, t=90, b=100), title="Box Plot by Sensor")
    style_fig(fig_box)
    img_b = render(fig_box, 1600, 1300)
    temp_files.append(img_b)

    dist_pair = Table(
        [[Image(img_h, width=HALF_W, height=SIDE_H+100), Image(img_b, width=HALF_W, height=SIDE_H+100)]],
        colWidths=[HALF_W + GAP/2, HALF_W + GAP/2])
    dist_pair.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(dist_pair)
    story.append(Paragraph(
        "Left: frequency distribution (red line = threshold). "
        "Right: box plot per sensor (wider = more variability).",
        caption_style))

    dist_parts = []
    overall_mean = filtered_df["Temperature"].mean()
    overall_std = filtered_df["Temperature"].std()
    overall_median = filtered_df["Temperature"].median()
    pct_above = (filtered_df["Temperature"] > THRESHOLD).mean() * 100
    dist_parts.append(f"Mean temperature: <b>{overall_mean:.1f}°C</b>, median: <b>{overall_median:.1f}°C</b>, std dev: <b>{overall_std:.1f}°C</b>.")
    if abs(overall_mean - overall_median) > 2:
        dist_parts.append("The mean-median gap indicates a <b>skewed distribution</b> — check for outliers.")
    else:
        dist_parts.append("Mean ≈ median suggests a <b>roughly symmetric distribution</b>.")
    dist_parts.append(f"<b>{pct_above:.1f}%</b> of all readings exceed the {THRESHOLD}°C threshold.")
    widest = sensor_stats.loc[(sensor_stats["Max Temp (°C)"] - sensor_stats["Min Temp (°C)"]).idxmax()]
    dist_parts.append(f"<b>Sensor {widest['Sensor Id']}</b> has the widest range ({widest['Min Temp (°C)']:.1f}–{widest['Max Temp (°C)']:.1f}°C).")
    story.append(make_takeaway_box(" ".join(dist_parts)))
    story.append(PageBreak())

    # ==================== TEMPORAL: HOURLY + DAILY ====================
    story.append(Paragraph("Temporal Patterns — Hourly & Daily", page_title))
    story.append(Paragraph(
        "When temperature and spike patterns change by hour and day of week.", page_desc))

    hourly_avg_pdf = filtered_df.groupby(filtered_df["time"].dt.hour)["Temperature"].mean().reset_index()
    hourly_avg_pdf.columns = ["Hour", "Avg Temperature"]
    fig_h1 = px.bar(hourly_avg_pdf, x="Hour", y="Avg Temperature",
        color="Avg Temperature", color_continuous_scale="RdYlGn_r",
        title="Avg Temperature by Hour",
        labels={"Hour": "Hour (0–23)", "Avg Temperature": "Avg Temp (°C)"})
    fig_h1.update_layout(template="plotly_white", xaxis=dict(dtick=2),
        margin=dict(l=100, r=60, t=90, b=80))
    style_fig(fig_h1)

    hr_spike = filtered_df[filtered_df["Is Spike"]].groupby(
        filtered_df[filtered_df["Is Spike"]]["time"].dt.hour).size().reset_index(name="Spike Count")
    hr_spike.columns = ["Hour", "Spike Count"]
    all_hrs = pd.DataFrame({"Hour": range(24)})
    hr_spike = all_hrs.merge(hr_spike, on="Hour", how="left").fillna(0)
    fig_h2 = px.bar(hr_spike, x="Hour", y="Spike Count",
        color="Spike Count", color_continuous_scale="YlOrRd",
        title="Spike Events by Hour",
        labels={"Hour": "Hour (0–23)", "Spike Count": "Spikes"})
    fig_h2.update_layout(template="plotly_white", xaxis=dict(dtick=2),
        margin=dict(l=100, r=60, t=90, b=80))
    style_fig(fig_h2)

    img_h1 = render(fig_h1, 1600, 1200)
    img_h2 = render(fig_h2, 1600, 1200)
    temp_files.extend([img_h1, img_h2])
    row1 = Table(
        [[Image(img_h1, width=HALF_W, height=SIDE_H), Image(img_h2, width=HALF_W, height=SIDE_H)]],
        colWidths=[HALF_W + GAP/2, HALF_W + GAP/2])
    row1.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(row1)

    peak_hv = hourly_avg_pdf.loc[hourly_avg_pdf["Avg Temperature"].idxmax()]
    h_insight = f"Peak temperature hour: {int(peak_hv['Hour'])}:00 ({peak_hv['Avg Temperature']:.1f}°C avg)."
    if hr_spike["Spike Count"].sum() > 0:
        pk_sp = hr_spike.loc[hr_spike["Spike Count"].idxmax()]
        h_insight += f" Most spikes at {int(pk_sp['Hour'])}:00 ({int(pk_sp['Spike Count'])} events)."
    story.append(Paragraph(h_insight, chart_note))
    story.append(Spacer(1, 20))

    day_order_pdf = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_df = filtered_df.copy()
    dow_df["DayOfWeek"] = dow_df["time"].dt.day_name()
    dow_t = dow_df.groupby("DayOfWeek")["Temperature"].mean().reset_index()
    dow_t.columns = ["Day", "Avg Temperature"]
    dow_t["Day"] = pd.Categorical(dow_t["Day"], categories=day_order_pdf, ordered=True)
    dow_t = dow_t.sort_values("Day")
    fig_d1 = px.bar(dow_t, x="Day", y="Avg Temperature",
        color="Avg Temperature", color_continuous_scale="RdYlGn_r",
        title="Avg Temperature by Day of Week")
    fig_d1.update_layout(template="plotly_white", margin=dict(l=100, r=60, t=90, b=80))
    style_fig(fig_d1)

    dow_sp = dow_df[dow_df["Is Spike"]].groupby("DayOfWeek").size().reset_index(name="Spike Count")
    all_days_p = pd.DataFrame({"DayOfWeek": day_order_pdf})
    dow_sp = all_days_p.merge(dow_sp, on="DayOfWeek", how="left").fillna(0)
    dow_sp["Spike Count"] = dow_sp["Spike Count"].astype(int)
    dow_sp["DayOfWeek"] = pd.Categorical(dow_sp["DayOfWeek"], categories=day_order_pdf, ordered=True)
    dow_sp = dow_sp.sort_values("DayOfWeek")
    fig_d2 = px.bar(dow_sp, x="DayOfWeek", y="Spike Count",
        color="Spike Count", color_continuous_scale="YlOrRd",
        title="Spike Events by Day of Week", labels={"DayOfWeek": "Day"})
    fig_d2.update_layout(template="plotly_white", margin=dict(l=100, r=60, t=90, b=80))
    style_fig(fig_d2)

    img_d1 = render(fig_d1, 1600, 1200)
    img_d2 = render(fig_d2, 1600, 1200)
    temp_files.extend([img_d1, img_d2])
    row2 = Table(
        [[Image(img_d1, width=HALF_W, height=SIDE_H), Image(img_d2, width=HALF_W, height=SIDE_H)]],
        colWidths=[HALF_W + GAP/2, HALF_W + GAP/2])
    row2.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(row2)

    if not dow_t.empty:
        pk_d = dow_t.loc[dow_t["Avg Temperature"].idxmax()]
        cl_d = dow_t.loc[dow_t["Avg Temperature"].idxmin()]
        d_insight = f"Hottest: {pk_d['Day']} ({pk_d['Avg Temperature']:.1f}°C). Coolest: {cl_d['Day']} ({cl_d['Avg Temperature']:.1f}°C)."
        story.append(Paragraph(d_insight, chart_note))

    temp_parts = []
    temp_parts.append(f"Peak temperature hour: <b>{int(peak_hv['Hour'])}:00</b> ({peak_hv['Avg Temperature']:.1f}°C avg).")
    if hr_spike["Spike Count"].sum() > 0:
        pk_sp = hr_spike.loc[hr_spike["Spike Count"].idxmax()]
        temp_parts.append(f"Most spikes occur at <b>{int(pk_sp['Hour'])}:00</b> ({int(pk_sp['Spike Count'])} events) — consider scheduling inspections around this time.")
    if not dow_t.empty:
        temp_parts.append(f"<b>{pk_d['Day']}</b> is the hottest day ({pk_d['Avg Temperature']:.1f}°C avg), <b>{cl_d['Day']}</b> is the coolest ({cl_d['Avg Temperature']:.1f}°C avg).")
        day_range = pk_d["Avg Temperature"] - cl_d["Avg Temperature"]
        if day_range > 5:
            temp_parts.append(f"Day-of-week variation is <b>{day_range:.1f}°C</b> — strong operational pattern detected.")
        else:
            temp_parts.append(f"Day-of-week variation is only <b>{day_range:.1f}°C</b> — consistent across the week.")
    story.append(make_takeaway_box(" ".join(temp_parts)))
    story.append(PageBreak())

    # ==================== TEMPORAL: HEATMAP + MONTHLY ====================
    story.append(Paragraph("Temporal Patterns — Heatmap & Monthly", page_title))

    tod_hm = filtered_df.copy()
    tod_hm["Label"] = tod_hm["Sensor Id"].map(lambda x: pdf_label(x))
    tod_hm_pivot = tod_hm.pivot_table(values="Temperature", index="Label",
        columns=tod_hm["time"].dt.hour, aggfunc="mean")
    tod_hm = tod_hm_pivot
    if not tod_hm.empty:
        fig_thm = px.imshow(tod_hm.values,
            x=[f"{h}:00" for h in tod_hm.columns],
            y=[str(s) for s in tod_hm.index],
            color_continuous_scale="RdYlGn_r",
            labels=dict(x="Hour of Day", y="Sensor Id", color="Avg Temp (°C)"),
            aspect="auto")
        fig_thm.update_layout(template="plotly_white",
            coloraxis_colorbar=dict(title="Avg Temp (°C)", len=0.8),
            margin=dict(l=120, r=80, t=60, b=100))
        style_fig(fig_thm)
        img = render(fig_thm, 3400, 1400)
        temp_files.append(img)
        story.append(Image(img, width=W, height=MAIN_H - 100))
        story.append(Paragraph(
            "Sensor × hour heatmap: warmer colors = hotter periods. Look for horizontal/vertical bands.",
            caption_style))
    story.append(Spacer(1, 20))

    mo_df = filtered_df.copy()
    mo_df["Month"] = mo_df["time"].dt.to_period("M").astype(str)
    m_t = mo_df.groupby("Month")["Temperature"].mean().reset_index()
    m_t.columns = ["Month", "Avg Temp"]
    fig_m1 = px.bar(m_t, x="Month", y="Avg Temp", color="Avg Temp",
        color_continuous_scale="RdYlGn_r", title="Avg Temperature per Month")
    fig_m1.update_layout(template="plotly_white", margin=dict(l=100, r=60, t=90, b=80))
    style_fig(fig_m1)

    m_sp = mo_df[mo_df["Is Spike"]].groupby("Month").size().reset_index(name="Spikes")
    if m_sp.empty:
        m_sp = pd.DataFrame({"Month": [], "Spikes": []})
    fig_m2 = px.bar(m_sp, x="Month", y="Spikes", color="Spikes",
        color_continuous_scale="YlOrRd", title="Spike Events per Month")
    fig_m2.update_layout(template="plotly_white", margin=dict(l=100, r=60, t=90, b=80))
    style_fig(fig_m2)

    if not alerts_df.empty:
        ma = alerts_df.copy()
        ma["Month"] = ma["Alert Time"].dt.to_period("M").astype(str)
        m_al = ma.groupby("Month").size().reset_index(name="Alerts")
    else:
        m_al = pd.DataFrame({"Month": [], "Alerts": []})
    fig_m3 = px.bar(m_al, x="Month", y="Alerts", color_discrete_sequence=[ACCENT],
        title="Alert Events per Month")
    fig_m3.update_layout(template="plotly_white", margin=dict(l=100, r=60, t=90, b=80))
    style_fig(fig_m3)

    THIRD_W = W * 0.31
    THIRD_GAP = W * 0.035
    img_m1 = render(fig_m1, 1200, 1000)
    img_m2 = render(fig_m2, 1200, 1000)
    img_m3 = render(fig_m3, 1200, 1000)
    temp_files.extend([img_m1, img_m2, img_m3])
    m_row = Table(
        [[Image(img_m1, width=THIRD_W, height=SIDE_H-100),
          Image(img_m2, width=THIRD_W, height=SIDE_H-100),
          Image(img_m3, width=THIRD_W, height=SIDE_H-100)]],
        colWidths=[THIRD_W + THIRD_GAP, THIRD_W + THIRD_GAP, THIRD_W + THIRD_GAP])
    m_row.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(m_row)

    if not m_t.empty:
        pk_m = m_t.loc[m_t["Avg Temp"].idxmax()]
        cl_m = m_t.loc[m_t["Avg Temp"].idxmin()]
        m_ins = f"Hottest month: {pk_m['Month']} ({pk_m['Avg Temp']:.1f}°C). Coolest: {cl_m['Month']} ({cl_m['Avg Temp']:.1f}°C)."
        if not m_sp.empty and len(m_sp) > 0 and "Spikes" in m_sp.columns and m_sp["Spikes"].sum() > 0:
            pk_ms = m_sp.loc[m_sp["Spikes"].idxmax()]
            m_ins += f" Most spikes in {pk_ms['Month']} ({int(pk_ms['Spikes'])} events)."
        story.append(Paragraph(m_ins, chart_note))

    tm_parts = []
    if not m_t.empty:
        tm_parts.append(f"Hottest month: <b>{pk_m['Month']}</b> ({pk_m['Avg Temp']:.1f}°C). Coolest: <b>{cl_m['Month']}</b> ({cl_m['Avg Temp']:.1f}°C).")
        month_range = pk_m["Avg Temp"] - cl_m["Avg Temp"]
        if month_range > 5:
            tm_parts.append(f"Monthly variation of <b>{month_range:.1f}°C</b> suggests <b>seasonal influence</b> on operating temperatures.")
        else:
            tm_parts.append(f"Monthly variation of only <b>{month_range:.1f}°C</b> — temperatures are <b>seasonally stable</b>.")
    if not m_al.empty and len(m_al) > 0 and "Alerts" in m_al.columns and m_al["Alerts"].sum() > 0:
        pk_al = m_al.loc[m_al["Alerts"].idxmax()]
        tm_parts.append(f"Month with most alert events: <b>{pk_al['Month']}</b> ({int(pk_al['Alerts'])} alerts).")
    if not tod_hm.empty:
        tm_parts.append("The sensor × hour heatmap above reveals which specific sensor-time combinations run hottest — target these for preventive maintenance.")
    story.append(make_takeaway_box(" ".join(tm_parts)))

    if has_legend and legend_df is not None:
        story.append(PageBreak())
        # ==================== SENSOR METADATA SUMMARY ====================
        story.append(Paragraph("Sensor Metadata Summary", page_title))
        story.append(Paragraph("Sensor configuration from Legend_Master sheet with threshold values and type classification.", page_desc))

        meta_hdr = [Paragraph("Legend", table_hdr), Paragraph("Sensor UID", table_hdr),
                     Paragraph("Type", table_hdr), Paragraph("Description", table_hdr),
                     Paragraph("Threshold (°C)", table_hdr)]
        meta_data = [meta_hdr]
        for _, lrow in legend_df.iterrows():
            leg = str(lrow["Legend"]) if pd.notna(lrow["Legend"]) else ""
            uid = str(lrow["Sensor UID"]) if pd.notna(lrow["Sensor UID"]) else ""
            desc = str(lrow["Description"]) if pd.notna(lrow["Description"]) else ""
            thresh_val = f"{lrow['Threshold Temp']:.0f}" if pd.notna(lrow["Threshold Temp"]) else "N/A"
            stype = derive_sensor_type(leg) if leg else "Other"
            meta_data.append([
                Paragraph(leg, table_cell_l), Paragraph(uid, table_cell),
                Paragraph(stype, table_cell), Paragraph(desc, table_cell_l),
                Paragraph(thresh_val, table_cell),
            ])
        meta_cw = [W*0.12, W*0.20, W*0.10, W*0.40, W*0.15]
        meta_tbl = Table(meta_data, colWidths=meta_cw)
        meta_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor(PRIMARY)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor(BG_CARD), HexColor(BG_LIGHT)]),
            ("BOX", (0, 0), (-1, -1), 2, HexColor(BORDER)),
            ("INNERGRID", (0, 0), (-1, -1), 1, HexColor(BORDER)),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(meta_tbl)
        story.append(Spacer(1, 30))

        type_counts = sensor_stats["Sensor Type"].value_counts()
        meta_insight_parts = [f"<b>{len(legend_df)} sensors</b> configured in Legend_Master."]
        for st_name, cnt in type_counts.items():
            meta_insight_parts.append(f"{st_name}: <b>{cnt}</b>.")
        unique_thresholds = sensor_stats["Sensor Threshold"].unique()
        if len(unique_thresholds) > 1:
            meta_insight_parts.append(f"Thresholds range from <b>{min(unique_thresholds):.0f}°C</b> to <b>{max(unique_thresholds):.0f}°C</b>.")
        else:
            meta_insight_parts.append(f"Uniform threshold of <b>{unique_thresholds[0]:.0f}°C</b> across all sensors.")
        story.append(make_takeaway_box(" ".join(meta_insight_parts)))
        story.append(PageBreak())

        # ==================== THRESHOLD ANALYSIS ====================
        story.append(Paragraph("Threshold Analysis", page_title))
        story.append(Paragraph("Per-sensor threshold breach analysis using uploaded threshold values.", page_desc))

        breach_chart_data = []
        for sid in selected_sensors:
            s_data = filtered_df[filtered_df["Sensor Id"] == sid]
            s_thresh = sensor_threshold_map.get(sid, THRESHOLD)
            breach_count = int((s_data["Temperature"] > s_thresh).sum())
            breach_chart_data.append({
                "Sensor": pdf_label(sid),
                "Threshold": s_thresh,
                "Breach Count": breach_count,
                "Max Temp": round(s_data["Temperature"].max(), 1),
            })
        breach_chart_df = pd.DataFrame(breach_chart_data)

        fig_breach_pdf = px.bar(breach_chart_df, x="Sensor", y="Breach Count",
            color="Breach Count", color_continuous_scale="Reds",
            title="Threshold Breach Count by Sensor")
        fig_breach_pdf.update_layout(template="plotly_white", xaxis_type="category",
            margin=dict(l=100, r=60, t=90, b=80))
        style_fig(fig_breach_pdf)

        fig_maxvt_pdf = go.Figure()
        fig_maxvt_pdf.add_trace(go.Bar(x=breach_chart_df["Sensor"], y=breach_chart_df["Max Temp"],
            name="Max Temperature", marker_color=BLUE))
        fig_maxvt_pdf.add_trace(go.Scatter(x=breach_chart_df["Sensor"], y=breach_chart_df["Threshold"],
            name="Threshold", mode="markers+lines", line=dict(color="red", width=3, dash="dash"),
            marker=dict(color="red", size=12)))
        fig_maxvt_pdf.update_layout(template="plotly_white", title="Max Temperature vs Per-Sensor Threshold",
            yaxis_title="Temperature (°C)", barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=100, r=60, t=90, b=80))
        style_fig(fig_maxvt_pdf)

        img_tb1 = render(fig_breach_pdf, 1600, 1200)
        img_tb2 = render(fig_maxvt_pdf, 1600, 1200)
        temp_files.extend([img_tb1, img_tb2])
        tb_pair = Table(
            [[Image(img_tb1, width=HALF_W, height=SIDE_H), Image(img_tb2, width=HALF_W, height=SIDE_H)]],
            colWidths=[HALF_W + GAP/2, HALF_W + GAP/2])
        tb_pair.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
        story.append(tb_pair)
        story.append(Spacer(1, 20))

        total_breaches = sum(b["Breach Count"] for b in breach_chart_data)
        worst_breach = max(breach_chart_data, key=lambda x: x["Breach Count"])
        tb_insight = [f"<b>{total_breaches}</b> total threshold breach readings across all sensors."]
        if worst_breach["Breach Count"] > 0:
            tb_insight.append(f"<b>{worst_breach['Sensor']}</b> has the most breaches ({worst_breach['Breach Count']} readings above {worst_breach['Threshold']}°C).")
        no_breach = [b for b in breach_chart_data if b["Breach Count"] == 0]
        if no_breach:
            tb_insight.append(f"<b>{len(no_breach)} sensor(s)</b> never exceeded their threshold — operating safely.")
        story.append(make_takeaway_box(" ".join(tb_insight)))
        story.append(PageBreak())

        # ==================== CORRELATION ANALYSIS ====================
        if len(selected_sensors) >= 2:
            story.append(Paragraph("Correlation Analysis", page_title))
            story.append(Paragraph("Cross-sensor temperature correlation and spike co-occurrence analysis.", page_desc))

            corr_pivot_pdf = filtered_df.pivot_table(values="Temperature", index="time",
                columns="Sensor Id", aggfunc="mean")
            corr_pivot_pdf.columns = [pdf_label(c) for c in corr_pivot_pdf.columns]
            corr_matrix_pdf = corr_pivot_pdf.corr()

            fig_corr_pdf = px.imshow(corr_matrix_pdf.values,
                x=list(corr_matrix_pdf.columns), y=list(corr_matrix_pdf.index),
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                labels=dict(color="Correlation"), aspect="auto")
            fig_corr_pdf.update_layout(template="plotly_white", title="Temperature Correlation Heatmap",
                margin=dict(l=120, r=80, t=90, b=100),
                coloraxis_colorbar=dict(title="Corr", len=0.8))
            style_fig(fig_corr_pdf)
            img_corr = render(fig_corr_pdf, 2400, 2000)
            temp_files.append(img_corr)
            story.append(Spacer(1, 20))

            type_corr_data_pdf = []
            for ci_idx, s1 in enumerate(selected_sensors):
                for cj_idx, s2 in enumerate(selected_sensors):
                    if ci_idx >= cj_idx:
                        continue
                    l1, l2 = pdf_label(s1), pdf_label(s2)
                    t1 = sensor_type_map.get(s1, "Other") if sensor_type_map else "Other"
                    t2 = sensor_type_map.get(s2, "Other") if sensor_type_map else "Other"
                    if l1 in corr_matrix_pdf.columns and l2 in corr_matrix_pdf.columns:
                        c_val = corr_matrix_pdf.loc[l1, l2]
                        comp_type = "Same Type" if t1 == t2 else "Cross Type"
                        type_corr_data_pdf.append({
                            "Pair": f"{l1} — {l2}",
                            "Correlation": round(c_val, 3),
                            "Comparison": comp_type,
                        })

            if type_corr_data_pdf:
                type_corr_df_pdf = pd.DataFrame(type_corr_data_pdf)
                fig_type_corr_pdf = px.bar(type_corr_df_pdf, x="Pair", y="Correlation", color="Comparison",
                    color_discrete_map={"Same Type": "#3498DB", "Cross Type": "#E8574A"},
                    labels={"Pair": "Sensor Pair", "Correlation": "Correlation Coefficient"})
                fig_type_corr_pdf.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
                fig_type_corr_pdf.update_layout(template="plotly_white", title="Same-Type vs Cross-Type Correlation",
                    xaxis=dict(tickangle=-45), margin=dict(l=80, r=40, t=90, b=140))
                style_fig(fig_type_corr_pdf)
                img_type_corr = render(fig_type_corr_pdf, 2400, 1400)
                temp_files.append(img_type_corr)

                corr_pair = Table(
                    [[Image(img_corr, width=HALF_W, height=MAIN_H),
                      Image(img_type_corr, width=HALF_W, height=MAIN_H)]],
                    colWidths=[HALF_W + GAP/2, HALF_W + GAP/2])
                corr_pair.setStyle(TableStyle([
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("LEFTPADDING", (0,0), (-1,-1), 4),
                    ("RIGHTPADDING", (0,0), (-1,-1), 4),
                ]))
                story.append(corr_pair)
                story.append(Paragraph(
                    "Left: Temperature correlation heatmap. Right: Same-type vs cross-type sensor pair correlations.",
                    caption_style))
            else:
                story.append(Image(img_corr, width=W*0.7, height=MAIN_H))
                story.append(Paragraph("Red = strong positive correlation. Blue = negative. White = no relationship.", caption_style))
            story.append(Spacer(1, 20))

            tm_sensors_pdf = [s for s in selected_sensors if (sensor_type_map or {}).get(s) == "TM"]
            non_tm_pdf = [s for s in selected_sensors if (sensor_type_map or {}).get(s) in ("MSU", "Axle")]
            if tm_sensors_pdf and non_tm_pdf:
                tm_comp_pdf = []
                for tm_s in tm_sensors_pdf:
                    tm_lbl = pdf_label(tm_s)
                    for other_s in non_tm_pdf:
                        other_lbl = pdf_label(other_s)
                        if tm_lbl in corr_matrix_pdf.columns and other_lbl in corr_matrix_pdf.columns:
                            c_val = corr_matrix_pdf.loc[tm_lbl, other_lbl]
                            tm_comp_pdf.append({
                                "TM Sensor": tm_lbl,
                                "Compared With": other_lbl,
                                "Type": (sensor_type_map or {}).get(other_s, "Other"),
                                "Correlation": round(c_val, 3),
                            })
                if tm_comp_pdf:
                    tm_comp_df_pdf = pd.DataFrame(tm_comp_pdf)
                    fig_tm_pdf = px.bar(tm_comp_df_pdf, x="Compared With", y="Correlation", color="TM Sensor",
                        barmode="group", labels={"Compared With": "MSU/Axle Sensor", "Correlation": "Correlation"},
                        color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_tm_pdf.update_layout(template="plotly_white", title="TM Sensors vs Nearby MSU/Axle Sensors",
                        margin=dict(l=80, r=40, t=90, b=80))
                    style_fig(fig_tm_pdf)
                    img_tm = render(fig_tm_pdf, 2400, 1200)
                    temp_files.append(img_tm)
                    story.append(Image(img_tm, width=W*0.75, height=SIDE_H))
                    story.append(Paragraph("TM sensor correlation with nearby MSU and Axle sensors.", caption_style))
                    story.append(Spacer(1, 20))

            spike_pivot_pdf = filtered_df.pivot_table(values="Is Spike", index="time",
                columns="Sensor Id", aggfunc="max").fillna(False).astype(int)
            spike_pivot_pdf.columns = [pdf_label(c) for c in spike_pivot_pdf.columns]
            if spike_pivot_pdf.shape[1] >= 2:
                co_occur_pdf = []
                sp_cols = list(spike_pivot_pdf.columns)
                for ci in range(len(sp_cols)):
                    for cj in range(ci+1, len(sp_cols)):
                        s1_sp = spike_pivot_pdf[sp_cols[ci]]
                        s2_sp = spike_pivot_pdf[sp_cols[cj]]
                        both = int((s1_sp & s2_sp).sum())
                        either = int((s1_sp | s2_sp).sum())
                        jaccard_val = round(both / either, 3) if either > 0 else 0
                        co_occur_pdf.append({
                            "Sensor A": sp_cols[ci],
                            "Sensor B": sp_cols[cj],
                            "Co-occurring Spikes": both,
                            "Jaccard Index": jaccard_val,
                        })
                co_occur_df_pdf = pd.DataFrame(co_occur_pdf).sort_values("Co-occurring Spikes", ascending=False)
                if co_occur_df_pdf["Co-occurring Spikes"].sum() > 0:
                    fig_cooccur_pdf = px.bar(co_occur_df_pdf.head(15), x="Sensor A", y="Co-occurring Spikes",
                        color="Sensor B", barmode="group",
                        labels={"Co-occurring Spikes": "Simultaneous Spikes"},
                        color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_cooccur_pdf.update_layout(template="plotly_white",
                        title="Spike Co-occurrence Analysis",
                        margin=dict(l=80, r=40, t=90, b=80))
                    style_fig(fig_cooccur_pdf)
                    img_cooccur = render(fig_cooccur_pdf, 2400, 1200)
                    temp_files.append(img_cooccur)
                    story.append(Image(img_cooccur, width=W*0.75, height=SIDE_H))
                    story.append(Paragraph(
                        "Sensor pairs that experience spikes at the same time. "
                        "The Jaccard Index measures overlap: 0 = never spike together, 1 = always spike together.",
                        caption_style))
                    story.append(Spacer(1, 10))

                    co_table_data = [["Sensor A", "Sensor B", "Co-occurring Spikes", "Jaccard Index"]]
                    for _, row in co_occur_df_pdf.head(10).iterrows():
                        co_table_data.append([
                            str(row["Sensor A"]), str(row["Sensor B"]),
                            str(int(row["Co-occurring Spikes"])), str(row["Jaccard Index"]),
                        ])
                    co_tbl = Table(co_table_data, colWidths=[W*0.25]*4)
                    co_tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0,0), (-1,0), HexColor(PRIMARY)),
                        ("TEXTCOLOR", (0,0), (-1,0), white),
                        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                        ("FONTSIZE", (0,0), (-1,-1), 26),
                        ("LEADING", (0,0), (-1,-1), 36),
                        ("ALIGN", (0,0), (-1,-1), "CENTER"),
                        ("GRID", (0,0), (-1,-1), 0.5, HexColor(BORDER)),
                        ("ROWBACKGROUNDS", (0,1), (-1,-1), [HexColor(BG_LIGHT), HexColor(BG_CARD)]),
                        ("TOPPADDING", (0,0), (-1,-1), 8),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                    ]))
                    story.append(co_tbl)
                    story.append(Spacer(1, 20))

            corr_vals = []
            cols_list_pdf = list(corr_matrix_pdf.columns)
            for ci in range(len(cols_list_pdf)):
                for cj in range(ci+1, len(cols_list_pdf)):
                    corr_vals.append({
                        "Pair": f"{cols_list_pdf[ci]} — {cols_list_pdf[cj]}",
                        "Correlation": round(corr_matrix_pdf.iloc[ci, cj], 3),
                    })
            if corr_vals:
                avg_corr = np.mean([c["Correlation"] for c in corr_vals])
                max_corr = max(corr_vals, key=lambda x: x["Correlation"])
                min_corr = min(corr_vals, key=lambda x: x["Correlation"])
                corr_insight = [f"Average cross-sensor correlation: <b>{avg_corr:.3f}</b>."]
                corr_insight.append(f"Strongest positive: <b>{max_corr['Pair']}</b> ({max_corr['Correlation']:.3f}).")
                corr_insight.append(f"Weakest/most negative: <b>{min_corr['Pair']}</b> ({min_corr['Correlation']:.3f}).")
                if avg_corr > 0.7:
                    corr_insight.append("High average correlation suggests sensors share common thermal influences.")
                elif avg_corr < 0.3:
                    corr_insight.append("Low average correlation suggests sensors operate independently.")
                story.append(make_takeaway_box(" ".join(corr_insight)))

    doc.build(story, onFirstPage=draw_page_header_footer, onLaterPages=draw_page_header_footer)

    for f in temp_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    buf.seek(0)
    return buf.getvalue()


if st.button("Generate PDF Report", type="primary"):
    with st.spinner("Generating professional A0 landscape PDF report..."):
        heatmap_fig_for_pdf = None
        if not heatmap_pivot.empty:
            heatmap_fig_for_pdf = fig_heatmap

        pdf_fig_combined = go.Figure()
        y_max_pdf = max(filtered_df["Temperature"].max(), 120)
        pdf_fig_combined.add_hrect(y0=0, y1=SAFE_LIMIT, fillcolor="green", opacity=0.07, line_width=0)
        pdf_fig_combined.add_hrect(y0=SAFE_LIMIT, y1=WARNING_LIMIT, fillcolor="orange", opacity=0.07, line_width=0)
        pdf_fig_combined.add_hrect(y0=WARNING_LIMIT, y1=y_max_pdf * 1.1, fillcolor="red", opacity=0.07, line_width=0)
        for i, sensor_id in enumerate(selected_sensors):
            sensor_data = filtered_df[filtered_df["Sensor Id"] == sensor_id]
            color = colors[i % len(colors)]
            label = get_label(sensor_id)
            pdf_fig_combined.add_trace(go.Scatter(x=sensor_data["time"], y=sensor_data["Temperature"],
                                                    mode="lines", name=label, line=dict(color=color, width=1.5), opacity=0.7))
            pdf_fig_combined.add_trace(go.Scatter(x=sensor_data["time"], y=sensor_data["Rolling Avg"],
                                                    mode="lines", name=f"{label} (Avg)",
                                                    line=dict(color=color, width=2.5, dash="dot"), opacity=0.9))
        if has_legend:
            unique_thresholds = set(sensor_threshold_map.get(s, THRESHOLD) for s in selected_sensors)
            for t_val in sorted(unique_thresholds):
                pdf_fig_combined.add_hline(y=t_val, line_dash="dash", line_color="red", line_width=2,
                                             annotation_text=f"Threshold ({t_val}°C)", annotation_position="top right")
        else:
            pdf_fig_combined.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", line_width=2,
                                         annotation_text=f"Threshold ({THRESHOLD}°C)", annotation_position="top right")
        pdf_fig_combined.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)", height=500, template="plotly_white",
                                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                         margin=dict(l=60, r=30, t=60, b=60))

        pdf_bytes = generate_pdf_report(
            filtered_df, alerts_df, sensor_stats, pdf_fig_combined,
            fig_hist, fig_box, fig_roc, heatmap_fig_for_pdf,
            selected_sensors, total_alerts, total_records,
            alert_pct, filtered_min_ts, filtered_max_ts,
            spike_count, insights,
            has_legend=has_legend, sensor_legend_map=sensor_legend_map,
            sensor_threshold_map=sensor_threshold_map, sensor_type_map=sensor_type_map,
            sensor_description_map=sensor_description_map, legend_df=legend_df,
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"temperature_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
        st.success("PDF report generated successfully! The report is in A0 landscape format, optimized for poster printing.")
