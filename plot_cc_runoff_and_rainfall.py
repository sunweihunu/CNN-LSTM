import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # 导入 MaxNLocator

# Define a threshold for significant rainfall (e.g., > 10mm)
rainfall_threshold = 100  # 10表示1mm

# Set the font to Times New Roman and font size to 20
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

# Load the original dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Initialize the lag range (0-20 rows)
max_lag = 20  # Up to 20 rows
lags_extended = np.arange(0, max_lag + 1)

# Define the precipitation columns for each station
cumulative_precipitation_stations = ['cz_prec', 'lz_prec', 'sg_prec']

# Calculate cumulative precipitation for each station at each lag (0-20 rows)
cumulative_prec_stations = {}
for station in cumulative_precipitation_stations:
    station_cumulative_precip = {}
    for lag in lags_extended:
        if lag == 0:
            # For lag=0, use the current precipitation value (no accumulation)
            cumulative_prec = data[station]
        else:
            # For lag>0, calculate cumulative precipitation
            cumulative_prec = data[station].rolling(window=lag + 1).sum()
        station_cumulative_precip[lag] = cumulative_prec
    cumulative_prec_stations[station] = pd.DataFrame(station_cumulative_precip)

# Initialize a dictionary to store filtered data for each lag
filtered_data_by_lag = {}

# For each lag, filter events where all three stations have cumulative precipitation > threshold
for lag in lags_extended:
    if rainfall_threshold > 0:
        # Check if all three stations have cumulative precipitation > threshold
        mask = (
                (cumulative_prec_stations['cz_prec'][lag] > rainfall_threshold) &
                (cumulative_prec_stations['lz_prec'][lag] > rainfall_threshold) &
                (cumulative_prec_stations['sg_prec'][lag] > rainfall_threshold)
        )
    else:
        # If rainfall_threshold is 0, include all data points
        mask = np.ones(len(data), dtype=bool)

    # Store the filtered data for the current lag
    filtered_data_by_lag[lag] = data[mask]

# Calculate lagged correlations for each station's cumulative precipitation with runoff
lagged_correlations = {station: {} for station in cumulative_precipitation_stations}

for lag in lags_extended:
    # Get the filtered data for the current lag
    filtered_data = filtered_data_by_lag[lag]
    # Calculate correlation for each station
    for station in cumulative_precipitation_stations:
        corr = cumulative_prec_stations[station][lag].corr(filtered_data['runoff_lin'])
        lagged_correlations[station][lag] = corr

# Convert the results to DataFrames for easier plotting
lagged_correlations_df = {
    station: pd.DataFrame(list(lagged_correlations[station].items()), columns=['Lag (rows)', 'Correlation'])
    for station in cumulative_precipitation_stations
}

# Convert lag rows to hours (1 row = 6 hours)
for station in lagged_correlations_df:
    lagged_correlations_df[station]['Lag (hours)'] = lagged_correlations_df[station]['Lag (rows)'] * 6

# Plot the lagged correlations for each station
fig, ax1 = plt.subplots(figsize=(12, 8))  # Increase figure size for better readability
dict_name = {
    'cz_prec': 'Chenzhou station',
    'lz_prec': 'Lianzhou station',
    'sg_prec': 'Shaoguan station',
}

# Plot correlation lines
for station in cumulative_precipitation_stations:
    ax1.plot(
        lagged_correlations_df[station]['Lag (hours)'],
        lagged_correlations_df[station]['Correlation'],
        linewidth=2,  # Increase line width
        marker='o',  # Add markers
        markersize=8,  # Set marker size
        label=f'{dict_name[station]}'
    )

# Add labels, legend, and title
ax1.set_title(f"Lagged Correlations (0-120 hours) for Cumulative Precipitation (>{int(rainfall_threshold / 10)} mm)",
              fontsize=22, pad=20)
ax1.set_xlabel("Lag (hours)", fontsize=20, labelpad=10)
ax1.set_ylabel("Correlation Coefficient", fontsize=20, labelpad=10)
ax1.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a horizontal line at y=0
ax1.grid(alpha=0.5, linestyle='--')  # Add grid lines with dashed style

# Adjust tick label font size
ax1.tick_params(axis='both', labelsize=20)

# Add a second y-axis for sample size
ax2 = ax1.twinx()
sample_sizes = [len(filtered_data_by_lag[lag]) for lag in lags_extended]
ax2.plot(
    lagged_correlations_df['cz_prec']['Lag (hours)'],
    sample_sizes,
    color='black',  # Use black color for the sample size line
    linestyle=':',  # Use a dotted line
    linewidth=4,  # Increase line width
    label='Sample Size for Calculation'
)

# Adjust the number of ticks on the right y-axis
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))  # 设置刻度数量为 6

# Set label for the second y-axis
ax2.set_ylabel("Sample Size for Calculation", fontsize=20, labelpad=10, color='black')
ax2.tick_params(axis='y', labelsize=20, colors='black')  # Adjust tick label color and size

# Combine legends for both axes and place them in the lower right corner
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='upper center',  # 将图例放在顶部居中
    bbox_to_anchor=(0.45, 1),  # 调整图例的位置（x, y）
    fontsize=20,
    ncol=2  # 将图例分为2列显示
)
# Set the left y-axis range
ax1.set_ylim(bottom=-0.2, top=1)

# Set the right y-axis range
ax2.set_ylim(bottom=-1190, top=6000)

print("Correlations for lag=0:")
print({station: lagged_correlations[station][0] for station in cumulative_precipitation_stations})

# Show plot
plt.tight_layout()  # Ensure all elements fit within the figure
plt.savefig(
    f'{int(rainfall_threshold/10)} mm_threshold_cc.svg',
    bbox_inches='tight',  # 自动裁剪空白区域
    pad_inches=0.1  # 设置较小的边距
)

# 初始化最大值和对应的站点及滞后时间
max_correlation = -1  # 假设相关性最小为 -1
max_station = None
max_lag = None

# 遍历所有站点和滞后时间
for station in cumulative_precipitation_stations:
    for lag, corr in lagged_correlations[station].items():
        if corr > max_correlation:
            max_correlation = corr
            max_station = station
            max_lag = lag

# 打印结果
print(f"Maximum correlation: {max_correlation:.4f} (Station: {max_station}, Lag: {max_lag*6})")