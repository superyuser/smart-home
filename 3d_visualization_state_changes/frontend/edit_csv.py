import pandas as pd

df = pd.read_csv('smoothed_output.csv')

# Rename columns
df = df.rename(columns={
    'attn': 'focus',
    'tired': 'fatigue',
    'anx': 'stress'
})

# Optional: normalize to 0-1 if needed
df['focus'] = df['focus'].clip(0, 1)
df['fatigue'] = df['fatigue'].clip(0, 1)
df['stress'] = df['stress'].clip(0, 1)

# Save new CSV
df[['focus', 'fatigue', 'stress']].to_csv('smoothed_labeled_output.csv', index=False)
