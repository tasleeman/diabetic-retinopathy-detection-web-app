import pandas as pd

# Read original CSV
df = pd.read_csv('train_labels.csv')

# Remove .png from all id_code entries
df['id_code'] = df['id_code'].str.replace('.png', '', regex=False)

# Save cleaned version
df.to_csv('train_labels_clean.csv', index=False)
print("Created train_labels_clean.csv")