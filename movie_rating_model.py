import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

class MovieRatingPredictor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.model = RandomForestRegressor()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.X = None
        self.y = None

    def load_data(self):
        self.df = pd.read_csv(self.dataset_path, encoding='latin1')
        self.df.columns = self.df.columns.str.strip().str.lower()
        print("Loaded columns:", self.df.columns.tolist())

    def preprocess(self):
        # Drop rows with missing ratings
        self.df.dropna(subset=['rating'], inplace=True)

        # Fill missing categorical values
        self.df.fillna({'genre': 'Unknown', 'director': 'Unknown',
                    'actor 1': 'Unknown', 'actor 2': 'Unknown', 'actor 3': 'Unknown'}, inplace=True)

        # Combine actors into single column
        self.df['actors'] = self.df['actor 1'] + "|" + self.df['actor 2'] + "|" + self.df['actor 3']

        # Categorical & numeric features
        self.df['votes'] = self.df['votes'].astype(str).str.replace(',', '', regex=False).astype(float)

        categorical = self.df[['genre', 'director', 'actors']]
        numeric = self.df[['votes']]
    
        encoded = self.encoder.fit_transform(categorical)
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out())

        self.X = pd.concat([numeric.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        self.y = self.df['rating']

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Model trained.\nMAE: {mae:.2f}\nR² Score: {r2:.2f}")

    def save_model(self):
        with open("model.pkl", "wb") as f:
            pickle.dump((self.model, self.encoder), f)
        print("Model saved as model.pkl")

    def visualize_data(self):
        print("\nShowing visualizations...\n")

        plt.figure(figsize=(6, 4))
        sns.barplot(data=self.df, x='genre', y='rating', estimator='mean')
        plt.title('Average Rating by Genre')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        top_directors = self.df.groupby('director')['rating'].mean().sort_values(ascending=False).head(5)
        top_directors.plot(kind='bar', figsize=(6, 4), title='Top 5 Directors by Avg Rating')
        plt.ylabel("Average Rating")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=self.df, x='votes', y='rating')
        plt.title('Votes vs Rating')
        plt.tight_layout()
        plt.show()

    def predict_custom_movie(self):
        print("\nEnter new movie details to predict rating:")

        genre = input("Genre: ")
        director = input("Director: ")
        actors = input("Actors (use | to separate names): ")
        votes = float(input("Total votes (approx): "))

        input_df = pd.DataFrame({
            'genre': [genre],
            'director': [director],
            'actors': [actors],
            'votes': [votes]
        })

        encoded_input = self.encoder.transform(input_df[['genre', 'director', 'actors']])
        encoded_df = pd.DataFrame(encoded_input, columns=self.encoder.get_feature_names_out())
        final_input = pd.concat([input_df[['votes']].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        prediction = self.model.predict(final_input)[0]
        print(f"\nPredicted Rating: {prediction:.2f}/10")

def main():
    print("Movie Rating Prediction Project\n")
    app = MovieRatingPredictor("movie_dataset.csv")
    app.load_data()
    app.preprocess()
    app.train()
    app.save_model()

    view = input("\nDo you want to view data visualizations? (y/n): ").lower()
    if view == 'y':
        app.visualize_data()

    predict = input("\nDo you want to predict a custom movie rating? (y/n): ").lower()
    if predict == 'y':
        app.predict_custom_movie()

if __name__ == "__main__":
    main()