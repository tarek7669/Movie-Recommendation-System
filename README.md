# Movie-Recommendation-System
A Movie Recommendation System Movie with Collaborative Filtering

This project is a movie recommendation system built using Python, Pandas, and the Pearson correlation coefficient. It utilizes the MovieLens dataset to recommend movies to users based on their historical preferences and ratings. If you're a movie enthusiast looking for personalized movie recommendations, this system is designed just for you!

## Introduction
Recommendation systems are widely used in today's world to help users discover new products, movies, or content based on their past interactions and preferences. This project focuses on building a movie recommendation system using collaborative filtering, a popular technique in recommendation systems.

## Dataset
We used the MovieLens dataset, which contains a vast amount of movie ratings data from users. You can find the dataset here. This dataset includes movie ratings, user information, and movie metadata.

## Recommendation Methodology
This system utilizes the Pearson correlation coefficient to find similarities between users. The Pearson correlation coefficient measures the linear correlation between two sets of data, in this case, user ratings. The higher the correlation, the more similar two users' preferences are.

Here's how the recommendation system works:

1. Calculate the Pearson correlation coefficient between the target user and all other users in the dataset.
2. Select the top N most similar users.
3. Recommend movies that these similar users have rated highly but the target user hasn't seen.

## Contributing
Contributions to this project are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.
