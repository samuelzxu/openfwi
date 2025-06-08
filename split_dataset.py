from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('split_files/generated_ds.csv')
print(len(data))
print(data.head())
data['strata'] = data['id'].apply(lambda x: x.split('/')[0])
print(data.value_counts('strata'))
train, val = train_test_split(data, test_size=0.05, random_state=42, stratify=data['strata'])
print(len(train), len(val))
train.to_csv('split_files/train_ds.csv', index=False)
val.to_csv('split_files/val_ds.csv', index=False)