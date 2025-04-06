import pandas as pd

# Load CSV
df = pd.read_csv('../../smart_wearable/processed_data/processed_data_20250405_141236.csv')

# Rename columns
df = df.rename(columns={
    'focus_probability': 'focus',
    'fatigue_probability': 'fatigue',
    'stress_probability': 'stress'
})

# Clip to [0, 1] just in case
df['focus'] = df['focus'].clip(0, 1)
df['fatigue'] = df['fatigue'].clip(0, 1)
df['stress'] = df['stress'].clip(0, 1)

# Normalize only the desired columns
df[['focus', 'fatigue', 'stress']] = (
    df[['focus', 'fatigue', 'stress']] - df[['focus', 'fatigue', 'stress']].min()
) / (
    df[['focus', 'fatigue', 'stress']].max() - df[['focus', 'fatigue', 'stress']].min()
)

# Save
df[['focus', 'fatigue', 'stress']].to_csv('trained_labeled_output.csv', index=False)

print("✅ Normalized and saved to trained_labeled_output.csv")
