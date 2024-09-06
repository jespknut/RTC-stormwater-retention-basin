import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    'font.size': 20,  # Use larger font size
    'axes.titlesize': 16,
    'axes.labelsize': 17,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 17,
    'lines.linewidth': 1.9,  # Thicker lines
    'lines.markersize': 6,  # Larger markers
    'figure.figsize': (8, 8),  # Larger figure size
    'savefig.dpi': 300,  # Higher resolution for saved figures
    'axes.grid': True,  # Enable grid lines
    'grid.alpha': 0.5,  # Make grid lines lighter
    'grid.linestyle': '--',  # Dashed grid lines
    'axes.prop_cycle': plt.cycler('color', ['blue', 'red', 'green', 'black'])  # Color cycle
})

# Constants
area_m2 = 5.6  # Area in square meters
daily_avg_total_consumption = 0.0317  # Example value, in cubic meters
usage_type = "Toilet"  # Example usage type
max_storage = 0.1  # Maximum storage capacity in cubic meters
initial_storage = 0.0  # Initial storage in cubic meters
max_discharge_flow = 0.01085  # Maximum discharge rate in cubic meters per hour
evaporation_rate = 0.001  # Evaporation rate in cubic meters per hour
ddf = 3.0  # Degree-day factor in mm/C/day

# Convert DDF to mm/C/hour for use in the simulation
ddf_hourly = ddf / 24  # mm/C/hour

# Load precipitation data from CSV file with semicolon as delimiter
precip_data = pd.read_csv('precipitation_temperature_data.csv', delimiter=';')

# Parse the 'time' column as datetime
precip_data['time'] = pd.to_datetime(precip_data['time'])

# Utility Functions
def rainfall_to_inflow(rainfall_mm, area_m2):
    # 1 mm of rainfall on 1 m² = 0.001 m³
    return rainfall_mm * area_m2 * 0.001  # m³/hour

# Add columns for inflow in m³/hour and initialize snow storage column
precip_data['snow_storage'] = 0

# Add a column for inflow in m³/hour
precip_data['inflow'] = precip_data['precipitation_mm'].apply(lambda mm: rainfall_to_inflow(mm, area_m2))

# Demand multipliers
monthly_multipliers = {
    1: {"Toilet": 0.93047966, "Washing_machine": 1.06892158},
    2: {"Toilet": 0.903043877, "Washing_machine": 0.854704814},
    3: {"Toilet": 0.905707999, "Washing_machine": 0.864743813},
    4: {"Toilet": 0.834618007, "Washing_machine": 0.763427159},
    5: {"Toilet": 0.878085261, "Washing_machine": 0.851924784},
    6: {"Toilet": 0.899491713, "Washing_machine": 1.074636084},
    7: {"Toilet": 0.741981341, "Washing_machine": 0.764817174},
    8: {"Toilet": 0.968198017, "Washing_machine": 0.992779643},
    9: {"Toilet": 1.274759001, "Washing_machine": 1.004826442},
    10: {"Toilet": 1.299857835, "Washing_machine": 1.155720298},
    11: {"Toilet": 1.257699274, "Washing_machine": 1.453492415},
    12: {"Toilet": 1.106078015, "Washing_machine": 1.150005792}
}

weekday_multipliers = {
    1: {"Toilet": 1.176691033, "Washing_machine": 0.97286957},  # Monday
    2: {"Toilet": 0.919263033, "Washing_machine": 0.82142154},  # Tuesday
    3: {"Toilet": 0.934518326, "Washing_machine": 0.796425567},  # Wednesday
    4: {"Toilet": 1.164773595, "Washing_machine": 1.112947252},  # Thursday
    5: {"Toilet": 0.902922128, "Washing_machine": 0.899729725},  # Friday
    6: {"Toilet": 0.903302902, "Washing_machine": 1.110942562},  # Saturday
    7: {"Toilet": 0.998528983, "Washing_machine": 1.285663785}   # Sunday
}

hourly_multipliers = {
    0: {"Toilet": 0.009487359, "Washing_machine": 0.000402604},
    1: {"Toilet": 0.005397193, "Washing_machine": 0.000129765},
    2: {"Toilet": 0.004229308, "Washing_machine": 8.89298E-05},
    3: {"Toilet": 0.003689075, "Washing_machine": 0.000186934},
    4: {"Toilet": 0.003164268, "Washing_machine": 0.000101029},
    5: {"Toilet": 0.003444367, "Washing_machine": 5.74716E-05},
    6: {"Toilet": 0.006185764, "Washing_machine": 0.00024864},
    7: {"Toilet": 0.012101109, "Washing_machine": 0.000741687},
    8: {"Toilet": 0.014082368, "Washing_machine": 0.002033286},
    9: {"Toilet": 0.01293717, "Washing_machine": 0.003441039},
    10: {"Toilet": 0.019090871, "Washing_machine": 0.005882072},
    11: {"Toilet": 0.012125913, "Washing_machine": 0.005196042},
    12: {"Toilet": 0.012575099, "Washing_machine": 0.005057505},
    13: {"Toilet": 0.0116613, "Washing_machine": 0.004727799},
    14: {"Toilet": 0.013057256, "Washing_machine": 0.004502752},
    15: {"Toilet": 0.011664929, "Washing_machine": 0.004286175},
    16: {"Toilet": 0.011690943, "Washing_machine": 0.004485511},
    17: {"Toilet": 0.011805886, "Washing_machine": 0.004484301},
    18: {"Toilet": 0.011250226, "Washing_machine": 0.004724169},
    19: {"Toilet": 0.011067527, "Washing_machine": 0.004543285},
    20: {"Toilet": 0.011581747, "Washing_machine": 0.00437087},
    21: {"Toilet": 0.013920843, "Washing_machine": 0.003728397},
    22: {"Toilet": 0.017815606, "Washing_machine": 0.002040848},
    23: {"Toilet": 0.017327097, "Washing_machine": 0.002136433}
}

# Sensitivity Analysis Function
def sensitivity_analysis_on_uncertainty(precip_data, max_storage, initial_storage, max_discharge_flow, evaporation_rate, ddf_hourly, forecast_control_function, uncertainty_levels, **control_params):
    """
    Perform sensitivity analysis on forecast uncertainty.

    Parameters:
    - precip_data (DataFrame): Precipitation data.
    - max_storage (float): Maximum storage capacity of the basin.
    - initial_storage (float): Initial storage in the basin.
    - max_discharge_flow (float): Maximum discharge rate.
    - evaporation_rate (float): Evaporation rate.
    - forecast_control_function (function): Control function that uses forecast data.
    - uncertainty_levels (list of float): Different levels of forecast uncertainty to test.
    - control_params (dict): Additional parameters for the control function.

    Returns:
    - results (list of dict): List of performance metrics for each uncertainty level.
    """
    results = []

    for uncertainty_scale in uncertainty_levels:
        print(f"Running simulation with forecast uncertainty scale: {uncertainty_scale}...")
        control_params['uncertainty_scale'] = uncertainty_scale

        # Run the simulation
        storage_levels, discharges, reuses, overflows, snowpack_depth = simulate_basin_with_control(
            precip_data, max_storage, initial_storage, max_discharge_flow, evaporation_rate, ddf_hourly, forecast_control_function, **control_params
        )

        # Calculate performance metrics
        metrics = calculate_performance_metrics(storage_levels, discharges, reuses, overflows, precip_data)
        metrics['uncertainty_scale'] = uncertainty_scale
        results.append(metrics)

        print(metrics)
        print("\n" + "-"*50 + "\n")
    
    return results


def calculate_demand(daily_avg_total_consumption, hour, weekday, month, usage_type):
    month_multiplier = monthly_multipliers[month][usage_type]
    weekday_multiplier = weekday_multipliers[weekday][usage_type]
    hour_multiplier = hourly_multipliers[hour][usage_type]
    combined_multiplier = month_multiplier * weekday_multiplier * hour_multiplier
    demand = combined_multiplier * daily_avg_total_consumption
    return demand

def simulate_precipitation_forecast(current_time_index, precip_data, forecast_hours, uncertainty_scale=0.2):
    end_index = min(current_time_index + forecast_hours, len(precip_data))
    baseline_forecast = precip_data['precipitation_mm'].iloc[current_time_index:end_index].values
    if end_index - current_time_index < forecast_hours:
        recent_average = precip_data['precipitation_mm'].iloc[max(0, current_time_index - 24):current_time_index].mean()
        baseline_forecast = np.concatenate([baseline_forecast, np.full(forecast_hours - len(baseline_forecast), recent_average)])
    noise = np.random.normal(0, uncertainty_scale * baseline_forecast, forecast_hours)
    forecast_with_uncertainty = baseline_forecast + noise
    forecast_with_uncertainty = np.maximum(forecast_with_uncertainty, 0)
    return forecast_with_uncertainty.tolist()

def calculate_combined_demand(daily_avg_total_consumption, hour, weekday, month, usage_types):
    total_demand = 0
    for usage_type in usage_types:
        month_multiplier = monthly_multipliers[month][usage_type]
        weekday_multiplier = weekday_multipliers[weekday][usage_type]
        hour_multiplier = hourly_multipliers[hour][usage_type]
        combined_multiplier = month_multiplier * weekday_multiplier * hour_multiplier
        demand = combined_multiplier * daily_avg_total_consumption
        total_demand += demand
    return total_demand

# Control Functions
def simple_control(storage, max_storage, inflow, reuse, max_discharge_flow, open_threshold, close_threshold, discharging):
    # Calculate the storage level after accounting for inflow and reuse, but before discharge
    projected_storage = storage + inflow - reuse
    fill_rate = projected_storage / max_storage

    # Determine discharge based on the projected fill rate
    if discharging:
        if fill_rate < close_threshold:
            discharge = 0  # Close discharge orifice if below close threshold
            discharging = False
        else:
            discharge = min(max_discharge_flow, projected_storage)  # Continue discharging
    else:
        if fill_rate > open_threshold:
            discharge = min(max_discharge_flow, projected_storage)  # Start discharging
            discharging = True
        else:
            discharge = 0  # No discharge if between thresholds

    discharge = max(0, min(discharge, max_discharge_flow))  # Ensure discharge is within bounds
    return discharge, discharging




def forecast_based_control_with_reuse(storage, max_storage, inflow, reuse, max_discharge_flow, forecasted_inflow_volume, available_capacity, forecast_hours, precip_data, current_time_index):
    future_reuse_demand = 0
    for i in range(1, forecast_hours + 1):
        if current_time_index + i < len(precip_data):
            future_time = precip_data['time'].iloc[current_time_index + i]
            hour = future_time.hour
            weekday = future_time.dayofweek + 1
            month = future_time.month
            future_reuse_demand += calculate_demand(daily_avg_total_consumption, hour, weekday, month, usage_type)
    adjusted_available_capacity = available_capacity - future_reuse_demand
    if forecasted_inflow_volume > adjusted_available_capacity:
        potential_discharge = storage + inflow - reuse
        discharge = min(max_discharge_flow, potential_discharge, potential_discharge - future_reuse_demand)
    else:
        discharge = 0
    discharge = max(discharge, 0)
    return discharge

def simulate_basin_with_control(precip_data, max_storage, initial_storage, max_discharge_flow, evaporation_rate, ddf_hourly, control_function, **control_params):
    storage = initial_storage
    snow_storage = 0
    storage_levels = []
    discharges = []
    reuses = []
    overflows = []
    snowpack_depth = []  # Track snowpack depth
    inflows = []
    discharging = control_params.get('discharging', False)  # Initialize discharging state

    for index, row in precip_data.iterrows():
        precipitation = row['precipitation_mm']
        air_temp = row['air_temperature']
        time = row['time']
        hour = time.hour
        weekday = time.dayofweek + 1
        month = time.month

        if air_temp < 0:
            snow_storage += rainfall_to_inflow(precipitation, area_m2)
            inflow = 0
        else:
            melt_rate = ddf_hourly * air_temp
            melt_volume = melt_rate * area_m2 * 0.001
            melt_volume = min(melt_volume, snow_storage)
            snow_storage -= melt_volume
            inflow = rainfall_to_inflow(precipitation, area_m2) + melt_volume

        inflows.append(inflow)
        precip_data.at[index, 'inflow'] = inflow
        precip_data.at[index, 'snow_storage'] = snow_storage

        # Track snowpack depth
        snow_depth = snow_storage / area_m2 * 1000  # Convert m³ to mm
        snowpack_depth.append(snow_depth)

        usage_types = ["Toilet", "Washing_machine"]
        reuse_demand = calculate_combined_demand(daily_avg_total_consumption, hour, weekday, month, usage_types)
        reuse = min(storage, reuse_demand)
        available_capacity = max_storage - storage

        if control_function == forecast_based_control_with_reuse:
            forecasted_precipitation = simulate_precipitation_forecast(index, precip_data, control_params['forecast_hours'], control_params['uncertainty_scale'])
            forecasted_inflow_volume = sum([rainfall_to_inflow(mm, area_m2) for mm in forecasted_precipitation])
            discharge = control_function(storage, max_storage, inflow, reuse, max_discharge_flow, forecasted_inflow_volume, available_capacity, control_params['forecast_hours'], precip_data, index)
        else:
            discharge, discharging = control_function(storage, max_storage, inflow, reuse, max_discharge_flow, control_params['open_threshold'], control_params['close_threshold'], discharging)

        storage += inflow - discharge - reuse - evaporation_rate
        
        # Ensure that storage level is not negative
        storage = max(storage, 0)

        if storage > max_storage:
            overflow = storage - max_storage
            storage = max_storage
        else:
            overflow = 0

        storage_levels.append(storage)
        discharges.append(discharge)
        reuses.append(reuse)
        overflows.append(overflow)

    return storage_levels, discharges, reuses, overflows, snowpack_depth


def calculate_performance_metrics(storage_levels, discharges, reuses, overflows, precip_data, baseline_overflow=0, epsilon=0.001):
    # Calculate total reuse demand
    total_reuse_demand = sum([
        calculate_demand(daily_avg_total_consumption, row['time'].hour, row['time'].dayofweek + 1, row['time'].month, usage_type)
        for _, row in precip_data.iterrows()
    ])
    
    # Calculate total actual reuse
    total_actual_reuse = sum(reuses)
    
    # Calculate the percentage of reuse demand met
    percent_reuse_demand_met = (total_actual_reuse / total_reuse_demand) * 100 if total_reuse_demand > 0 else 0
    
    # Calculate total overflow volume
    total_overflow_volume = sum(overflows)
    
    # Overflow Score: Normalize RTC overflow compared to baseline overflow (Stupid Basin)
    # Using epsilon to avoid division by zero and ensure stability in calculations
    normalized_overflow = min((baseline_overflow + epsilon) / (total_overflow_volume + epsilon), 1)
    
    # Reuse Score: Normalize reuse based on total demand met
    normalized_reuse = total_actual_reuse / total_reuse_demand if total_reuse_demand > 0 else 0
    
    # Weights for each metric: Adjust these based on the scenario
    weight_overflow = 0.7  # Higher weight for overflow control
    weight_reuse = 0.3     # Weight for reuse (adjust as needed)

    # Calculate Weighted Scores
    weighted_overflow_score = weight_overflow * normalized_overflow
    weighted_reuse_score = weight_reuse * normalized_reuse
    
    # Total Performance Score (normalized and weighted)
    total_performance_score = weighted_overflow_score + weighted_reuse_score
    
    # Print detailed metrics for transparency
    print(f"Total Reuse Demand Met: {percent_reuse_demand_met:.2f}%")
    print(f"Total Overflow Volume: {total_overflow_volume:.2f} m³")
    print(f"Baseline Overflow Volume: {baseline_overflow:.2f} m³")
    print(f"Total Reused Volume: {total_actual_reuse:.2f} m³")
    print(f"Normalized Overflow: {normalized_overflow:.2f}")
    print(f"Normalized Reuse: {normalized_reuse:.2f}")
    print(f"Weighted Overflow Score: {weighted_overflow_score:.2f}")
    print(f"Weighted Reuse Score: {weighted_reuse_score:.2f}")
    print(f"Overall Performance Score: {total_performance_score:.2f}")
    
    # Return metrics as a dictionary
    return {
        "percent_reuse_demand_met": percent_reuse_demand_met,
        "total_overflow_volume": total_overflow_volume,
        "total_reused_volume": total_actual_reuse,
        "normalized_overflow": normalized_overflow,
        "normalized_reuse": normalized_reuse,
        "total_performance_score": total_performance_score
    }


# Plotting function with date range
def plot_results(precip_data, storage_levels_simple, storage_levels_forecast_reuse, discharges_simple, discharges_forecast_reuse, reuses_simple, reuses_forecast_reuse, overflows_simple, overflows_forecast_reuse, snowpack_depth, start_date=None, end_date=None):
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        mask = (precip_data['time'] >= start_date) & (precip_data['time'] <= end_date)
        time_filtered = precip_data.loc[mask, 'time']
        precip_filtered = precip_data.loc[mask, 'precipitation_mm']
        temp_filtered = precip_data.loc[mask, 'air_temperature']
        storage_simple_filtered = np.array(storage_levels_simple)[mask]
        storage_forecast_reuse_filtered = np.array(storage_levels_forecast_reuse)[mask]
        discharges_simple_filtered = np.array(discharges_simple)[mask]
        discharges_forecast_reuse_filtered = np.array(discharges_forecast_reuse)[mask]
        reuses_simple_filtered = np.array(reuses_simple)[mask]
        reuses_forecast_reuse_filtered = np.array(reuses_forecast_reuse)[mask]
        overflows_simple_filtered = np.array(overflows_simple)[mask]
        overflows_forecast_reuse_filtered = np.array(overflows_forecast_reuse)[mask]
        snowpack_depth_filtered = np.array(snowpack_depth)[mask]
    else:
        time_filtered = precip_data['time']
        precip_filtered = precip_data['precipitation_mm']
        temp_filtered = precip_data['air_temperature']
        storage_simple_filtered = storage_levels_simple
        storage_forecast_reuse_filtered = storage_levels_forecast_reuse
        discharges_simple_filtered = discharges_simple
        discharges_forecast_reuse_filtered = discharges_forecast_reuse
        reuses_simple_filtered = reuses_simple
        reuses_forecast_reuse_filtered = reuses_forecast_reuse
        overflows_simple_filtered = overflows_simple
        overflows_forecast_reuse_filtered = overflows_forecast_reuse
        snowpack_depth_filtered = snowpack_depth

    fig, axs = plt.subplots(5, 1, figsize=(7.5, 20), sharex=True)  # Adjusted figure size
    
    # Combined plot for Precipitation, Snowpack Depth, and Air Temperature
    ax1 = axs[0]
    ax1.plot(time_filtered, precip_filtered, label='Precipitation (mm)', linestyle='-', color='blue')
    ax1.set_ylabel('Precipitation (mm)')
    ax1.set_title('Precipitation, Snowpack Depth, and Air Temperature')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(time_filtered, temp_filtered, label='Air Temperature (°C)', linestyle=':', color='grey')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend(loc='upper right')
    
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(time_filtered, snowpack_depth_filtered, label='Snowpack Depth (mm)', linestyle='-', color='orange')
    ax3.set_ylabel('Snowpack Depth (mm)')
    ax3.legend(loc='upper center')
    
    # Storage Levels
    axs[1].plot(time_filtered, storage_simple_filtered, label='Rule-based Control', linestyle='-', color='cyan')
    axs[1].plot(time_filtered, storage_forecast_reuse_filtered, label='Forecast-Based Control', linestyle=':', color='black')
    axs[1].set_ylabel('Storage (m³)')
    axs[1].set_title('Storage Levels in the Detention Basin')
    
    
    # Discharges
    axs[2].plot(time_filtered, discharges_simple_filtered, label='Rule-based Control', linestyle='-', color='cyan')
    axs[2].plot(time_filtered, discharges_forecast_reuse_filtered, label='Forecast-Based Control', linestyle=':', color='black')
    axs[2].set_ylabel('Discharge (m³/h)')
    axs[2].set_title('Discharge')
    
    
    # Reuses
    axs[3].plot(time_filtered, reuses_simple_filtered, label='Rule-based Control', linestyle='-', color='cyan')
    axs[3].plot(time_filtered, reuses_forecast_reuse_filtered, label='Forecast-Based Control', linestyle=':', color='black')
    axs[3].set_ylabel('Reuse (m³)')
    axs[3].set_title('Water Reused ')
   
    
    # Overflows
    axs[4].plot(time_filtered, overflows_simple_filtered, label='Rule-based Control', linestyle='-', color='cyan')
    axs[4].plot(time_filtered, overflows_forecast_reuse_filtered, label='Forecast-Based Control', linestyle=':', color='black')
    axs[4].set_ylabel('Overflow (m³)')
    axs[4].set_title('Overflow')
    axs[4].legend()

    plt.tight_layout(pad=3.0)  # Add padding between plots
    plt.subplots_adjust(hspace=0.4, left=0.05, right=0.9)  # Adjust the height between subplots to avoid overlap
    plt.show()

# Define the plotting function for sensitivity analysis
def plot_sensitivity_analysis(sensitivity_results):
    uncertainty_scales = [result['uncertainty_scale'] for result in sensitivity_results]
    reuse_percentages = [result['percent_reuse_demand_met'] for result in sensitivity_results]
    overflow_volumes = [result['total_overflow_volume'] for result in sensitivity_results]
    performance_scores = [result['total_performance_score'] for result in sensitivity_results]

    plt.figure(figsize=(18, 6))

    # Plot Reuse Percentage
    plt.subplot(1, 3, 1)
    plt.plot(uncertainty_scales, reuse_percentages, marker='o', linestyle='-')
    plt.xlabel('Forecast Uncertainty Scale')
    plt.ylabel('Reuse Demand Met (%)')
    plt.title('Impact of Forecast Uncertainty on Reuse')

    # Plot Overflow Volume
    plt.subplot(1, 3, 2)
    plt.plot(uncertainty_scales, overflow_volumes, marker='o', linestyle='-')
    plt.xlabel('Forecast Uncertainty Scale')
    plt.ylabel('Total Overflow Volume (m³)')
    plt.title('Impact of Forecast Uncertainty on Overflow')

    # Plot Performance Score
    plt.subplot(1, 3, 3)
    plt.plot(uncertainty_scales, performance_scores, marker='o', linestyle='-')
    plt.xlabel('Forecast Uncertainty Scale')
    plt.ylabel('Performance Score')
    plt.title('Impact of Forecast Uncertainty on Performance')

    plt.tight_layout(pad=3.0)  # Add padding between plots
    plt.subplots_adjust(hspace=0.4)  # Adjust the height between subplots to avoid overlap
    plt.show()
    
def generate_csv_summary(results, filename='simulation_summary.csv'):
    """
    Generate a CSV summary of the simulation results.

    Parameters:
    - results (list of dict): List of performance metrics for each simulation.
    - filename (str): The name of the output CSV file.

    Returns:
    - None
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(df.to_csv(index=False))


# Define control parameters for forecast-based control with reuse consideration
forecast_control_params_with_reuse = {
    'forecast_hours': 24,
    'uncertainty_scale': 0.2  # Level of uncertainty in the forecast
}

# Threshold values for simple control
threshold_values = [
    {'open_threshold': 0.9, 'close_threshold': 0.8},
    {'open_threshold': 0.9, 'close_threshold': 0.4},
    {'open_threshold': 0.8, 'close_threshold': 0.2},
    {'open_threshold': 0.7, 'close_threshold': 0.3},
    {'open_threshold': 0.01, 'close_threshold': 0.00}
]

# Collect results for different simple control threshold values
simple_control_results = []

# Initialize the discharging state
discharging = False

# Run the simulation with simple control for different threshold values
for thresholds in threshold_values:
    print(f"Running simulation with simple control (open > {thresholds['open_threshold']*100}%, close < {thresholds['close_threshold']*100}%)...")
    storage_levels_simple, discharges_simple, reuses_simple, overflows_simple, snowpack_depth = simulate_basin_with_control(
        precip_data, max_storage, initial_storage, max_discharge_flow, evaporation_rate, ddf_hourly, simple_control,
        open_threshold=thresholds['open_threshold'], close_threshold=thresholds['close_threshold'], discharging=discharging
    )

    metrics_simple_control = calculate_performance_metrics(storage_levels_simple, discharges_simple, reuses_simple, overflows_simple, precip_data)
    metrics_simple_control['open_threshold'] = thresholds['open_threshold']
    metrics_simple_control['close_threshold'] = thresholds['close_threshold']
    simple_control_results.append(metrics_simple_control)
    print(metrics_simple_control)
    print("\n" + "-"*50 + "\n")

# Generate CSV summary for simple control results
#generate_csv_summary(simple_control_results, filename='simple_control_summary.csv')

# Collect results for forecast-based control
forecast_control_results = []

# Run the simulation with forecast-based control considering future reuse demand
print("Running simulation with forecast-based control considering future reuse demand...")
storage_levels_forecast_reuse, discharges_forecast_reuse, reuses_forecast_reuse, overflows_forecast_reuse, snowpack_depth = simulate_basin_with_control(
    precip_data, max_storage, initial_storage, max_discharge_flow, evaporation_rate, ddf_hourly, forecast_based_control_with_reuse, **forecast_control_params_with_reuse
)

# Save the simulation results to a CSV file
forecast_output_df = pd.DataFrame({
    'time': precip_data['time'],
    'storage_level': storage_levels_forecast_reuse,
    'reused_water': reuses_forecast_reuse,
    'reuse_demand': [calculate_combined_demand(daily_avg_total_consumption, row['time'].hour, row['time'].dayofweek + 1, row['time'].month, ["Toilet", "Washing_machine"]) for _, row in precip_data.iterrows()],
    'discharge': discharges_forecast_reuse,
    'overflow': overflows_forecast_reuse
})
forecast_output_df.to_csv('forecast_control_output_forecast.csv', index=False, sep=';')

metrics_forecast_control_reuse = calculate_performance_metrics(storage_levels_forecast_reuse, discharges_forecast_reuse, reuses_forecast_reuse, overflows_forecast_reuse, precip_data)
metrics_forecast_control_reuse['control_type'] = 'forecast_based'
forecast_control_results.append(metrics_forecast_control_reuse)
print(metrics_forecast_control_reuse)

# Generate CSV summary for forecast-based control results
generate_csv_summary(forecast_control_results, filename='forecast_control_summary.csv')


# Plotting Results for the Entire Dataset
plot_results(precip_data, storage_levels_simple, storage_levels_forecast_reuse, discharges_simple, discharges_forecast_reuse, reuses_simple, reuses_forecast_reuse, overflows_simple, overflows_forecast_reuse, snowpack_depth)

# Plotting Results for a Specific Date Range (Zoom In)
start_date = '2024-06-25'
end_date = '2024-07-25'
plot_results(precip_data, storage_levels_simple, storage_levels_forecast_reuse, discharges_simple, discharges_forecast_reuse, reuses_simple, reuses_forecast_reuse, overflows_simple, overflows_forecast_reuse, snowpack_depth, start_date, end_date)


# Define the range of uncertainty scales to test
uncertainty_levels = [ 0.1 ]  # Example values representing different levels of forecast uncertainty

# Perform the sensitivity analysis
sensitivity_results = sensitivity_analysis_on_uncertainty(
    precip_data, max_storage, initial_storage, max_discharge_flow, evaporation_rate, ddf_hourly, forecast_based_control_with_reuse, uncertainty_levels, forecast_hours=24
)

# Generate CSV summary for sensitivity analysis results
#generate_csv_summary(sensitivity_results, filename='sensitivity_analysis_summary.csv')

# Plot the sensitivity analysis results
#plot_sensitivity_analysis(sensitivity_results)
