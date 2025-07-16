# Movie Rating Prediction

<q>This project predicts the rating of a movie based on its genre, director, actors, and votes using machine learning. It was created as part of the CodSoft Internship and helps you explore real-world data analysis, preprocessing, and regression modeling.</q>

# Project Files
<ol>
<li><b>movie_rating_model.py</b> – Main Python file with the complete logic (loading data, preprocessing, training, predicting, and visualizing).</li>

<li><b>movie_rating_model.py</b> – Main Python file with the complete logic (loading data, preprocessing, training, predicting, and visualizing).</li>

<li><b>movie_dataset.csv</b> – Sample movie dataset containing columns like Genre, Director, Actors, Votes, and Rating.</li>

<li><b>model.pkl</b> – Trained regression model saved using pickle.</li>

<li><b>requirements.txt</b> – List of all Python libraries used in the project.</li>
</ol>

# Features
<ol>
<li>Cleaned and prepared dataset from scratch</li>

<li>Encodes categorical columns like Genre, Director, and Actors</li>

<li>Combines multiple actor columns into one</li>

<li>Trains a regression model using Random Forest</li>

<li>Displays Mean Absolute Error and R² score</li>

<li>Shows data visualizations (genre vs rating, top directors, votes vs rating)</li>

<li>Allows user to input custom movie details to predict its rating</li>

</ol>

# Visualizations

<b>The project displays:</b>
<ul>

<li>Average rating by genre</li>

<li>Top 5 directors by average rating</li>

<li>Scatter plot of votes vs rating</li>
</ul>

# How to Run

<b>Make sure all required libraries are installed:</b>

bash
Copy
Edit
pip install -r requirements.txt

<b>Run the main file:</b>

bash
Copy
Edit
<q>python movie_rating_model.py</q>

Follow the prompts in the terminal to:

<q>View visualizations</q>

Predict a custom movie rating

# Requirements

The project uses the following Python libraries:
<ul>
pandas
numpy
matplotlib
seaborn
scikit-learn
pickle
</ul>
