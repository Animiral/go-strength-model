#!/usr/bin/env python3
# Calculate mean and stdev of training labels in the list file given as the first program argument.

import sys
import pandas as pd

file_path = sys.argv[1]
df = pd.read_csv(file_path)

# Filter rows where Set == "T"
filtered_df = df[df['Set'] == "T"]

# Get the WhiteRating and BlackRating values
white_ratings = filtered_df['WhiteRating']
black_ratings = filtered_df['BlackRating']

# Combine the ratings
combined_ratings = pd.concat([white_ratings, black_ratings])

# Compute the mean and standard deviation
mean_rating = combined_ratings.mean()
stdev_rating = combined_ratings.std()

# import ace_tools as tools; tools.display_dataframe_to_user(name="Filtered Ratings Data", dataframe=filtered_df)

print(mean_rating, stdev_rating)
