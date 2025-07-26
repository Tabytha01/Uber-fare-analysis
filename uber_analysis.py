import pandas as pd
import numpy as np

# Load dataset
file_path = "uber.xlsx"
df = pd.read_excel(file_path)

print("First 5 rows:")
print(df.head())

print("\nColumns in the dataset:")
print(df.columns)

# --- Data Cleaning ---

# Drop rows with missing fare_amount or pickup_datetime
df_clean = df.dropna(subset=['fare_amount', 'pickup_datetime'])

# Remove invalid fares
df_clean = df_clean[df_clean['fare_amount'] > 0]

# Convert pickup_datetime to datetime
df_clean['pickup_datetime'] = pd.to_datetime(df_clean['pickup_datetime'], utc=True)

# Haversine formula to calculate trip distance in km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

df_clean['trip_distance'] = haversine(
    df_clean['pickup_latitude'], df_clean['pickup_longitude'],
    df_clean['dropoff_latitude'], df_clean['dropoff_longitude']
)

# Remove invalid trip distances
df_clean = df_clean[df_clean['trip_distance'] > 0]

# Remove fare_amount outliers (IQR)
Q1_fare = df_clean['fare_amount'].quantile(0.25)
Q3_fare = df_clean['fare_amount'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
lower_fare = Q1_fare - 1.5 * IQR_fare
upper_fare = Q3_fare + 1.5 * IQR_fare
df_clean = df_clean[(df_clean['fare_amount'] >= lower_fare) & (df_clean['fare_amount'] <= upper_fare)]

# Remove trip_distance outliers (IQR)
Q1_dist = df_clean['trip_distance'].quantile(0.25)
Q3_dist = df_clean['trip_distance'].quantile(0.75)
IQR_dist = Q3_dist - Q1_dist
lower_dist = Q1_dist - 1.5 * IQR_dist
upper_dist = Q3_dist + 1.5 * IQR_dist
df_clean = df_clean[(df_clean['trip_distance'] >= lower_dist) & (df_clean['trip_distance'] <= upper_dist)]

# --- Feature Engineering ---

df_clean['hour'] = df_clean['pickup_datetime'].dt.hour
df_clean['day'] = df_clean['pickup_datetime'].dt.day
df_clean['month'] = df_clean['pickup_datetime'].dt.month
df_clean['year'] = df_clean['pickup_datetime'].dt.year
df_clean['day_of_week'] = df_clean['pickup_datetime'].dt.day_name()

# Define peak hours (example: 7-9 AM and 4-7 PM)
df_clean['is_peak'] = df_clean['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)

# Save the final cleaned + feature-engineered dataset
df_clean.to_csv("uber_cleaned_features.csv", index=False)

print("\nAll done! Saved 'uber_cleaned_features.csv' with shape:", df_clean.shape)


import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# 1. Fare amount distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_clean['fare_amount'], bins=50, kde=True)
plt.title('Fare Amount Distribution')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.boxplot(x=df_clean['fare_amount'])
plt.title('Fare Amount Boxplot')

plt.tight_layout()
plt.show()

# 2. Scatter plot: Fare amount vs. Trip distance
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x='trip_distance', y='fare_amount', alpha=0.3)
plt.title('Fare Amount vs. Trip Distance')
plt.xlabel('Trip Distance (km)')
plt.ylabel('Fare Amount ($)')
plt.show()

# 3. Ride counts by hour of day
plt.figure(figsize=(10, 6))
hour_counts = df_clean['hour'].value_counts().sort_index()
sns.barplot(x=hour_counts.index, y=hour_counts.values, palette='viridis')
plt.title('Number of Rides by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Rides')
plt.show()

# 4. Ride counts by day of week
plt.figure(figsize=(10, 6))
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df_clean['day_of_week'].value_counts().reindex(order)
sns.barplot(x=day_counts.index, y=day_counts.values, palette='magma')
plt.title('Number of Rides by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Rides')
plt.show()

# 5. Average fare by hour
plt.figure(figsize=(10, 6))
avg_fare_hour = df_clean.groupby('hour')['fare_amount'].mean()
sns.lineplot(x=avg_fare_hour.index, y=avg_fare_hour.values, marker='o')
plt.title('Average Fare Amount by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Fare ($)')
plt.xticks(range(0, 24))
plt.grid(True)
plt.savefig("avg_fare_by_hour.png")

plt.show()
df.to_csv("uber_cleaned_features.csv", index=False)

