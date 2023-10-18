# Movie-Recommendation-System

This repository contains code for building a Collaborative Filtering Recommender System. Collaborative filtering is a popular technique used in recommendation systems to provide personalized movie recommendations based on user preferences and interactions.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Finding Similar Movies](#finding-similar-movies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Collaborative filtering is a powerful method for building recommendation systems. This code demonstrates how to create a collaborative filtering recommender system using a neural network. It includes steps for data preprocessing, model configuration, training, and making recommendations for both new and existing users.

## Prerequisites

Before using this code, you'll need the following prerequisites:

- Python 3
- Libraries: NumPy, Pandas, TensorFlow, Matplotlib, scikit-learn
- Movie dataset (provided in the code)

## Usage

To use this code and build your own collaborative filtering recommender system, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/tarek7669/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```

2. Ensure you have the required Python libraries installed:

   ```bash
   pip install numpy pandas tensorflow matplotlib scikit-learn
   ```

3. Prepare your movie dataset and place it in the appropriate location in the code.

4. Follow the code's comments and sections to understand how to load data, preprocess it, configure the neural network, train the model, make predictions for new and existing users, and find similar movies.

## Dataset

This code uses a movie dataset, which is provided in the code. However, you can replace it with your own dataset. The code reads and processes this dataset to make movie recommendations.

## Training the Model

The code trains a collaborative filtering model using a neural network. It uses user and item data to predict movie ratings. The model is configured with layers, loss functions, and optimizers to optimize the training process.

## Making Predictions

This code allows you to make predictions for both new and existing users. For new users, you can provide genre preferences and receive movie recommendations. For existing users, the model predicts their ratings for movies.

## Finding Similar Movies

One of the highlights of this code is the ability to find similar movies. The script computes the squared distance between movie feature vectors and identifies movies that are similar based on these vectors. It provides a table of similar movies to help users discover content they may like.

## Contributing

Contributions to this project are welcome! If you have ideas for improvements, bug fixes, or additional features, please feel free to open an issue or submit a pull request.

