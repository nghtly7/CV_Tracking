import pandas as pd
import matplotlib.pyplot as plt

# Load the tracking results
results = pd.read_csv('/Users/davide/Desktop/CV_Tracking/results/out2_tracking.csv')

# Display the first few rows
print(results.head())

# Basic statistics
print(results.describe())

# Count unique objects tracked
print(f"Number of unique objects: {results['id'].nunique()}")

# Visualize trajectories
plt.figure(figsize=(10, 8))
for obj_id in results['id'].unique():
    obj_data = results[results['id'] == obj_id]
    plt.plot(obj_data['x'], obj_data['y'], '-o', markersize=2, label=f'Object {obj_id}')

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Object Trajectories')
plt.legend()
plt.grid(True)
plt.show()