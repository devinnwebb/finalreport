import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the dataset
data = pd.read_csv("enriched_movies.csv")

# Preprocessing: Fill missing values explicitly
data = data.copy()  # Avoid chained assignment issues
data["vote_average"] = data["vote_average"].fillna(0)  
data["popularity"] = data["popularity"].fillna(0)      
data["overview"] = data["overview"].fillna("")         
data["release_date"] = data["release_date"].fillna("1900-01-01")  
data["genre_ids"] = data["genre_ids"].apply(lambda x: eval(x) if pd.notnull(x) else [])  
# Preprocess genres for similarity calculation
def preprocess_genres(data):
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(data["genre_ids"])
    return genre_matrix

# Normalize numeric features (ratings, popularity, and release date)
scaler = MinMaxScaler()
data["vote_average_scaled"] = scaler.fit_transform(data[["vote_average"]])
data["popularity_scaled"] = scaler.fit_transform(data[["popularity"]])

# Convert release_date to a numeric value (year)
data["release_year"] = pd.to_datetime(data["release_date"]).dt.year
data["release_year_scaled"] = scaler.fit_transform(data[["release_year"]])

# Preprocess overview text with TF-IDF
def preprocess_overviews(data):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(data["overview"])
    return tfidf_matrix

# Combine features for similarity (weighted scoring)
def calculate_similarity_matrix(data):
    genre_matrix = preprocess_genres(data)
    ratings = data["vote_average_scaled"].values.reshape(-1, 1)
    popularity = data["popularity_scaled"].values.reshape(-1, 1)
    release_year = data["release_year_scaled"].values.reshape(-1, 1)
    overview_matrix = preprocess_overviews(data)

    # Calculate similarities
    genre_similarity = cosine_similarity(genre_matrix)
    rating_similarity = cosine_similarity(ratings)
    popularity_similarity = cosine_similarity(popularity)
    overview_similarity = cosine_similarity(overview_matrix)
    release_similarity = cosine_similarity(release_year)

    # Weighted combination of similarities
    combined_similarity = (
        0.3 * genre_similarity +
        0.2 * rating_similarity +
        0.15 * popularity_similarity +
        0.3 * overview_similarity +
        0.05 * release_similarity
    )
    return combined_similarity

# Calculate similarity matrix
similarity_matrix = calculate_similarity_matrix(data)

# Recommend similar hidden gems for a specific title
def recommend_hidden_gems(movie_title, data, similarity_matrix, top_n=10):
    if movie_title not in data["title"].values:
        return f"Movie '{movie_title}' not found in dataset."
    
    # Find the movie index
    idx = data[data["title"] == movie_title].index[0]

    # Get similarity scores for the movie
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Define hidden gem criteria
    RATING_THRESHOLD = 7.5
    HARD_POPULARITY_THRESHOLD = 1.5
    SOFT_POPULARITY_THRESHOLD = 10.0

    # Filter recommendations for hidden gems
    hidden_gem_recommendations = []
    for i, _ in similarity_scores[1:]:
        title = data.iloc[i]["title"]
        rating = data.iloc[i]["vote_average"]
        popularity = data.iloc[i]["popularity"]
        if rating > RATING_THRESHOLD and popularity <= SOFT_POPULARITY_THRESHOLD:
            hidden_gem_recommendations.append((title, rating, popularity))
            if len(hidden_gem_recommendations) >= top_n:
                break
    return hidden_gem_recommendations


chosen_movie = "Scooby-Doo"  # Change the movie title here
hidden_gem_recommendations = recommend_hidden_gems(
    chosen_movie, data, similarity_matrix, top_n=10
)

# Display results
if isinstance(hidden_gem_recommendations, str):  # If error message is returned
    print(hidden_gem_recommendations)
else:
    print(f"\nHidden Gem Recommendations for '{chosen_movie}':")
    for title, rating, popularity in hidden_gem_recommendations:
        print(f"{title} (Rating: {rating}, Popularity: {popularity})")
