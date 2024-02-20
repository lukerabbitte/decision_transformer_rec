import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('goodreads/goodreads_data_1024_users.tsv', sep='\t')

train_data = pd.DataFrame()
test_data = pd.DataFrame()

for user in df['user_id'].unique():
    user_data = df[df['user_id'] == user]
    train, test = train_test_split(user_data, test_size=0.2)
    train_data = pd.concat([train_data, train])
    test_data = pd.concat([test_data, test])

train_data = train_data.reset_index(drop=True)
print(train_data.head())
test_data = test_data.reset_index(drop=True)
print(test_data.head())

train_data.to_csv('goodreads/goodreads_train_data_1024_users.tsv', sep='\t', index=False)
test_data.to_csv('goodreads/goodreads_test_data_1024_users.tsv', sep='\t', index=False)
