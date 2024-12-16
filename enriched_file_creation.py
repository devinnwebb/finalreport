import requests
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
import time

# TMDb API Configuration
API_URL = "Placeholder"
HEADERS = {
   
}

CACHE_FILE = "cache.json"

# Load or initialize the cache
try:
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
except FileNotFoundError:
    cache = {}

# Fetch movie details using TMDb API
def fetch_movie_details(imdb_id):
    if imdb_id in cache:  # Check cache
        return imdb_id, cache[imdb_id]

    url = f"{API_URL}{imdb_id}"
    params = {"external_source": "imdb_id"}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["movie_results"]:
            movie = data["movie_results"][0]
            details = {
                "imdb_id": imdb_id,
                "title": movie.get("title"),
                "overview": movie.get("overview"),
                "genre_ids": movie.get("genre_ids", []),
                "vote_average": movie.get("vote_average"),
                "release_date": movie.get("release_date"),
                "popularity": movie.get("popularity"),
                "poster_path": movie.get("poster_path"),
            }
            cache[imdb_id] = details  # Update cache
            return imdb_id, details
    return imdb_id, None

# Save cache periodically
def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# Enrich dataset with movie details
def enrich_dataset_parallel(data):
    enriched_data = []
    imdb_ids = data["imdb_id"].tolist()
    total = len(imdb_ids)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:  
        futures = executor.map(fetch_movie_details, imdb_ids)

        for i, (imdb_id, details) in enumerate(futures, start=1):
            enriched_row = {"imdb_id": imdb_id}
            if details:
                enriched_row.update(details)
            else:
                enriched_row.update({
                    "title": None,
                    "overview": None,
                    "genre_ids": None,
                    "vote_average": None,
                    "release_date": None,
                    "popularity": None,
                    "poster_path": None,
                })
            enriched_data.append(enriched_row)

            # Timer and progress
            elapsed_time = time.time() - start_time
            avg_time_per_item = elapsed_time / i
            remaining_time = avg_time_per_item * (total - i)
            print(f"Processed {i}/{total} | Estimated time remaining: {remaining_time:.2f} seconds", end="\r")

    save_cache()  # Save cache after processing
    return pd.DataFrame(enriched_data)

# Main function
def main():
    # Load dataset
    data = pd.read_json("imdb_movies_2000to2022.prolific (1).json", lines=True)

   
    data = data.drop_duplicates(subset=["imdb_id"])

    
    enriched_data = enrich_dataset_parallel(data)

    enriched_data.to_csv("enriched_movies.csv", index=False)
    print("\nEnrichment complete! Results saved to 'enriched_movies.csv'.")

if __name__ == "__main__":
    main()
