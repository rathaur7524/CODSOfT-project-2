# Movie Rating Prediction

This project predicts the rating of a movie based on its genre, director, actors, and votes using machine learning. It was created as part of the CodSoft Internship and helps you explore real-world data analysis, preprocessing, and regression modeling.

# Project Files

movie_rating_model.py – Main Python file with the complete logic (loading data, preprocessing, training, predicting, and visualizing).

movie_dataset.csv – Sample movie dataset containing columns like Genre, Director, Actors, Votes, and Rating.

model.pkl – Trained regression model saved using pickle.

requirements.txt – List of all Python libraries used in the project.

# Features

Cleaned and prepared dataset from scratch

Encodes categorical columns like Genre, Director, and Actors

Combines multiple actor columns into one

Trains a regression model using Random Forest

Displays Mean Absolute Error and R² score

Shows data visualizations (genre vs rating, top directors, votes vs rating)

Allows user to input custom movie details to predict its rating

# Visualizations

The project displays:

Average rating by genre

Top 5 directors by average rating

Scatter plot of votes vs rating

# How to Run

Make sure all required libraries are installed:

bash
Copy
Edit
pip install -r requirements.txt
Run the main file:

bash
Copy
Edit
python movie_rating_model.py
Follow the prompts in the terminal to:

View visualizations

Predict a custom movie rating

# Requirements

The project uses the following Python libraries:

nginx
Copy
Edit
pandas
numpy
matplotlib
seaborn
scikit-learn
pickle

All are listed in requirements.txt.