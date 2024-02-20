import pandas as pd

df = pd.read_csv('goodreads/goodreads_test_data_1024_users.tsv', sep='\t')

df_sorted = df.sort_values(by=['user_id', 'timestep'])

df_sorted.to_csv('goodreads/goodreads_test_data_1024_users_timestep_sorted.tsv', sep='\t', index=False)

print(df_sorted.head(60))