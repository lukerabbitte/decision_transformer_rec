import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import os

max_ratings_per_user = 273
min_ratings_per_user = 20

a, b = 1, 12  # shape parameters
data = np.floor(beta.rvs(a, b, size=10000)*(max_ratings_per_user-min_ratings_per_user+1)+min_ratings_per_user).astype(int)

print(np.mean(data))

# Show the histogram
plt.hist(data, bins=range(min_ratings_per_user, max_ratings_per_user+1), edgecolor='black')
plt.title('Distribution of number of items rated per user')
plt.xlabel('Number of items rated')
plt.ylabel('Number of users')

# Specify the directory
output_directory = 'figs'

# Check if the directory exists, and create it if not
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Save the plot as a PNG file in the specified directory
# plt.savefig(os.path.join(output_directory, 'distribution_of_synthetic_goodreads_reviews.png'))

# Show the plot (optional)
plt.show()
