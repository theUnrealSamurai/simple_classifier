import numpy as np
import pandas as pd
import random

# Set random seed for reproducibility
np.random.seed(42)

# Generate time series data
num_samples = 1000
time = pd.date_range('2024-01-01', periods=num_samples, freq='H')

# Generate motor data
motor_voltage = np.random.normal(loc=220, scale=5, size=num_samples)
motor_current = np.random.normal(loc=20, scale=2, size=num_samples)

# Generate capacitor data
capacitor_voltage = np.random.normal(loc=400, scale=10, size=num_samples)
capacitor_current = np.random.normal(loc=5, scale=1, size=num_samples)

# Generate other elements or parts data
other_elements = {
    'Temperature': np.random.normal(loc=25, scale=5, size=num_samples),
    'Pressure': np.random.normal(loc=100, scale=10, size=num_samples),
    'Humidity': np.random.normal(loc=50, scale=10, size=num_samples)
}

# Introduce anomalies
# Anomalies in motor voltage
motor_voltage[700:800] += np.random.normal(loc=100, scale=20, size=100)

# Anomalies in capacitor voltage
capacitor_voltage[500:600] += np.random.normal(loc=150, scale=30, size=100)

# Anomalies in other elements
other_elements['Temperature'][300:400] += np.random.normal(loc=50, scale=10, size=100)
other_elements['Pressure'][800:900] += np.random.normal(loc=150, scale=20, size=100)
other_elements['Humidity'][200:300] -= np.random.normal(loc=20, scale=5, size=100)

# Create DataFrame
data = pd.DataFrame({
    'Time': time,
    'Motor Voltage': motor_voltage,
    'Motor Current': motor_current,
    'Capacitor Voltage': capacitor_voltage,
    'Capacitor Current': capacitor_current,
    **other_elements
})

# Save to CSV file
data.to_csv('anamoly_detection/motor_data.csv', index=False)
