import pandas as pd
import numpy as np
import random
from scipy.stats import beta
import os


def create_synthetic_review_dataset(mu_file="goodreads/mu_goodreads4.csv", sigma_file="goodreads/sigma_goodreads4.csv",
                                    users_per_group=64, min_ratings_per_user=273, max_ratings_per_user=273):
    """
    Create a synthetic review dataset based on mean and standard deviation files.

    Parameters:
    - mu_file (str): File path for the mean values CSV file.
    - sigma_file (str): File path for the standard deviation values CSV file.
    - users_per_group (int): Number of users to generate within each group.
    - min_ratings_per_user (int): Minimum number of ratings per user. Follows ml100k baseline to avoid cold start.
    - max_ratings_per_user (int): Maximum number of ratings per user.

    Notes:
    - In ml100k dataset, ratio of users to items is 943/1682. To keep this ratio we have 153 users.
    - In ml100k dataset, least active user had rated 20/1682 items (~1%)
    - In ml100k dataset, most active user had rated 737/1682 items (~44%)
    - In ml100k dataset, average user had rated 106/1682 items (~7%)
    - We try and maintain similar ratios with our data

    Returns:
    - np arrays of states, actions, returns, terminal indices, returns-to-go, timesteps
    """

    # Set up dataframes and note shape
    means = pd.read_csv(mu_file, header=None)
    variances = pd.read_csv(sigma_file, header=None)
    num_items = means.shape[1]
    groups = means.shape[0]

    # The value at [group_index][item_index] gives us the mean and variance in their respective datasets.
    def get_rating(group_index, item_index):
        mean = -means.iloc[group_index, item_index]
        variance = variances.iloc[group_index, item_index]
        std_dev = np.sqrt(variance)
        rating = np.random.normal(mean, std_dev)
        rounded_rating = round(rating)
        clipped_rating = np.clip(rounded_rating, 1, 5)
        return round(clipped_rating)

    def generate_num_items_to_rate(a=1, b=12):
        """
        Generate number of items to rate based on beta distribution with shape a,b.
        A lower a and higher b skews more left.

        Returns:
        - Number between min_ratings_per_user and max_ratings_per_user pulled from left-skewed beta distribution
        """
        return int(np.floor(beta.rvs(a, b, size=1) * (max_ratings_per_user - min_ratings_per_user + 1) + min_ratings_per_user))

    def __generate_synthetic_data():
        """
        Generate synthetic MovieLens-style dataframe with the following columns:
        ['user_id', 'item_id', 'rating', 'timestep']

        Returns:
        - DataFrame containing synthetic user-item ratings data.
        """
        user_id = 1
        data = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestep'])

        for group in range(groups):
            for _ in range(users_per_group):
                num_items_to_rate = generate_num_items_to_rate()
                item_ids = random.sample(list(range(num_items)), num_items_to_rate)

                for timestep, item_id in enumerate(item_ids, start=1):
                    new_row = {'user_id': user_id, 'item_id': item_id + 1,
                               'rating': get_rating(group, item_id), 'timestep': timestep}
                    data.loc[len(data)] = new_row

                user_id += 1

        # Define the base filename
        base_filename = 'goodreads/goodreads_data.tsv'

        # Check if the file already exists
        if os.path.exists(base_filename):
            # Add a suffix to the filename until a non-existing filename is found
            suffix = 1
            while os.path.exists(f"{base_filename}_{suffix}.tsv"):
                suffix += 1
            filename_with_suffix = f"{base_filename}_{suffix}.tsv"
        else:
            filename_with_suffix = base_filename

        # Save the DataFrame to the file with the chosen filename
        data.to_csv(filename_with_suffix, sep='\t', index=False)

        return data

    __generate_synthetic_data()


if __name__ == "__main__":
    create_synthetic_review_dataset()