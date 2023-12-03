import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

project_start = datetime(2023, 11, 6)

adjusted_tasks_data = {
    "Task": ["Research", "Antenna Design", "Simulation",
             "Prototype Fabrication", "Testing Setup", "Data Collection",
             "Data Analysis", "Validation", "Final Reporting"],
    "Start": [project_start,
              project_start + timedelta(weeks=1),
              project_start + timedelta(weeks=3),
              project_start + timedelta(weeks=8),
              project_start + timedelta(weeks=9),
              project_start + timedelta(weeks=10),
              project_start + timedelta(weeks=11),
              project_start + timedelta(weeks=13),
              project_start + timedelta(weeks=14)],
    "Duration": [7*7, 14, 14, 14, 7, 14, 14, 7, 21]
    }

# Create a DataFrame with the adjusted task data
adjusted_df = pd.DataFrame(adjusted_tasks_data)

# Calculate the finish dates with the adjusted durations
adjusted_df['Finish'] = adjusted_df.apply(lambda row: row['Start'] + timedelta(days=row['Duration']), axis=1)

# Plotting the adjusted Gantt chart
fig, ax = plt.subplots(figsize=(14, 8))

# Creating the Gantt chart with adjusted tasks
for i, task in adjusted_df.iterrows():
    start = task['Start']
    finish = task['Finish']
    ax.barh(task['Task'], (finish - start).days, left=(start - project_start).days, height=0.4, align='center', color='skyblue', edgecolor='black')
    ax.text((start - project_start).days + (finish - start).days / 2, i, str((finish - start).days) + ' days', va='center', ha='center', color='black', fontsize=8)

# Set the range of x-ticks to cover the 4-month period
days_range = (adjusted_df['Finish'].max() - project_start).days
ax.set_xticks(range(0, days_range, 7))
ax.set_xticklabels([f"{dt.strftime('%d %B')}" for dt in [project_start + timedelta(weeks=i) for i in range(17)]])

# Formatting the chart
ax.set_xlabel('Timeline (Weeks)')
ax.set_title('Project Gantt Chart')
plt.gca().invert_yaxis()
plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5)

# Improve layout
plt.tight_layout()

# Show the adjusted Gantt chart
plt.show()
