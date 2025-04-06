states = ['Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Emergency', 'Emergency', 'Emergency', 'Emergency', 'Emergency', 'Emergency', 'Emergency', 'Fatigue', 'Fatigue', 'Fatigue', 'Emergency', 'Fatigue', 'Fatigue', 'Fatigue', 'Fatigue', 'Emergency', 'Emergency', 'Emergency', 'Emergency', 'Emergency', 'Neutral', 'Emergency', 'Emergency', 'Fatigue', 'Fatigue', 'Emergency', 'Fatigue', 'Fatigue', 'Fatigue', 'Fatigue', 'Fatigue', 'Fatigue', 'Fatigue', 'Fatigue', 'Emergency', 'Emergency', 'Emergency', 'Neutral']

import csv

all_data = []
with open("./smoothed_output.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        all_data.append(row)
    
print(len(all_data))

time_needed = 41
print(f"Time needed: {time_needed} seconds")
print(f"Num states: {len(states)}")