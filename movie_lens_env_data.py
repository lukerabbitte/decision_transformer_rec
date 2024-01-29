import pandas as pd
from utils.utils import get_terminal_indices

# File to use as env testing data
filepath = 'ml-100k/u1.test'

# Define column headers
column_headers = ['user_id', 'movie_id', 'rating', 'timestamp']

# Create a DataFrame from the CSV file with column headers
df = pd.read_csv(filepath, sep='\t', header=None, names=column_headers)

# Calculate the count of reviews per user
user_reviews_count = df['user_id'].value_counts()

# Only consider reviews where the user has reviewed at least 10
df_filtered = df[df['user_id'].isin(user_reviews_count[user_reviews_count >= 10].index)]

# # Save filtered reviews to new file
# df_filtered.to_csv('ml-100k/u1.test.greater_than_10.tsv', sep='\t', index=False)

print(df_filtered.iloc[136])

states = df_filtered['user_id'].tolist()
states_done_idxs = get_terminal_indices(states)

print(states_done_idxs)
