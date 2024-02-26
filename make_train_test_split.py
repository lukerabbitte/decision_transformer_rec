import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestep'])

train_data = pd.DataFrame()
test_data = pd.DataFrame()

for user in df['user_id'].unique():
    user_data = df[df['user_id'] == user]
    train, test = train_test_split(user_data, test_size=0.2)
    train_data = pd.concat([train_data, train])
    test_data = pd.concat([test_data, test])

# Sort timesteps and make cumulative for each dataset, one-indexed
train_data = train_data.reset_index(drop=True)
train_data = train_data.sort_values(by=['user_id', 'timestep'])
train_data['timestep'] = train_data.groupby('user_id').cumcount() + 1

test_data = test_data.reset_index(drop=True)
test_data = test_data.sort_values(by=['user_id', 'timestep'])
test_data['timestep'] = test_data.groupby('user_id').cumcount() + 1

train_data.to_csv('ml-100k/u.data_train', sep='\t', index=False)
test_data.to_csv('ml-100k/u.data_test', sep='\t', index=False)
