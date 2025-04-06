

filepath = "processed_data/processed_data_20250405_141236.csv"
import csv
all_states = []

with open(filepath, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        state = row[6]
        if state.lower() != "unknown":
            all_states.append(state)

print(all_states)
    