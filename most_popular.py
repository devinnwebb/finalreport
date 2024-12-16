import pandas as pd

# Load the dataset
file_path = "enriched_movies.csv"  
data = pd.read_csv(file_path)

# Ensure the 'popularity' column exists and is numeric
data['popularity'] = pd.to_numeric(data['popularity'], errors='coerce')
data = data.dropna(subset=['popularity'])

# Calculate thresholds
mean_popularity = data['popularity'].mean()
median_popularity = data['popularity'].median()
percentile_25 = data['popularity'].quantile(0.25)
percentile_10 = data['popularity'].quantile(0.10)

# Display thresholds
print(f"Mean Popularity: {mean_popularity:.2f}")
print(f"Median Popularity: {median_popularity:.2f}")
print(f"25th Percentile Popularity (Low Popularity): {percentile_25:.2f}")
print(f"10th Percentile Popularity (Very Low Popularity): {percentile_10:.2f}")
