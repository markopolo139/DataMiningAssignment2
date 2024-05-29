import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

# Load data from a CSV file
def load_data(path):
    return pd.read_csv(path)

# Drop specified columns from the DataFrame
def drop_columns(data, column):
    data.drop(columns=column, inplace=True)
    
# Drop rows with missing values
def drop_na(data):
    data.dropna(inplace=True)

# Drop duplicate rows
def drop_duplicate(data):
    data.drop_duplicates(inplace=True)

# Merge two DataFrames on specified columns with a given join method
def merge_data(data1, data2, on, how):
    return pd.merge(data1, data2, on=on, how=how)

# Calculate the average of a specified column grouped by another column
def calculate_average(data, groupby_column, average_column):
    average = data.groupby(groupby_column)[average_column].mean().reset_index()
    average.columns = [groupby_column, 'average_' + average_column]
    return average

# Create a deep copy of the DataFrame
def copy_data(data):
    return copy.deepcopy(data)

# Transform a categorical attribute to multiple binary attributes
def transform_attribute_to_multiple(data, attribute, splitted_by):
    list = []
    for index, row in data.iterrows():
        sublist = row[attribute].split(splitted_by)
        for item in sublist:
            list.append(item)
    
    for item in list:
        data[item] = 0
    
    for index, row in data.iterrows():
        sublist = row[attribute].split(splitted_by)
        for item in sublist:
            data.at[index, item] = 1
    
    return data

# Convert categorical string attribute to numeric values
def transform_strings_to_numbers(data, attribute):
    list = []

    for index, row in data.iterrows():
        if row[attribute] not in list:
            list.append(row[attribute])

    for i in range(len(list)):
        data.loc[data[attribute] == list[i], attribute] = i + 1
    
    return data

# Normalize specified columns using Min-Max scaling
def normalize_data(data, columns):
    normalize_data = copy_data(data)
    scaler = MinMaxScaler()
    normalize_data[columns] = scaler.fit_transform(normalize_data[columns])
    return normalize_data

# Standardize specified columns using Standard scaling
def standardize_data(data, columns):
    standardize_data = copy_data(data)
    scaler = StandardScaler()
    standardize_data[columns] = scaler.fit_transform(standardize_data[columns])
    return standardize_data

# Split data into training and testing sets
def split_data(data):
    return train_test_split(data, test_size=0.2, random_state=42)

# Round a number to the nearest half
def round_to_neares_half(number):
    return round(number * 2) / 2

# Rule-based rating prediction based on user genre preferences
def rule_based_rating(user_id, movie_id, data, genre_columns):
    user_data = data[data['userId'] == user_id]
    movie_data = data[data['movieId'] == movie_id]
    
    if user_data.empty or movie_data.empty:
        return None
    
    user_genre_preferences = user_data[genre_columns].mean()
    movie_genres = movie_data[genre_columns].iloc[0]
    
    score = (user_genre_preferences * movie_genres).sum()
    average_user_rating = user_data['rating'].mean()
    
    rating = score + average_user_rating
    return rating

# Clustering-based rating prediction using KMeans
def clustering_based_rating(user_id, movie_id, train_data, test_data, genre_columns, numerical_features):
    user_features = train_data[['userId'] + numerical_features + genre_columns].drop_duplicates()
    
    if user_features.empty:
        return None
    
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(user_features.drop(columns=['userId']))
    
    user_data = user_features[user_features['userId'] == user_id]
    if user_data.empty:
        return None
    
    user_cluster = kmeans.predict(user_data.drop(columns=['userId']))
    cluster_center = kmeans.cluster_centers_[user_cluster]
    
    movie_features = train_data[['movieId'] + numerical_features + genre_columns].drop_duplicates()
    if movie_features.empty:
        return None
    
    movie_data = movie_features[movie_features['movieId'] == movie_id]
    if movie_data.empty:
        return None
    
    movie_features['cluster_distance'] = pairwise_distances_argmin_min(movie_features.drop(columns=['movieId']), cluster_center)[0]
    
    closest_movie = movie_features.loc[movie_features['cluster_distance'].idxmin()]
    predicted_rating = closest_movie['rating']
    return predicted_rating

# Combined rating prediction using rule-based and clustering-based methods
def combined_rating(user_id, movie_id, train_data, test_data, scaler, genre_columns, numerical_features):
    rule_rating = rule_based_rating(user_id, movie_id, train_data, genre_columns)
    clustering_rating = clustering_based_rating(user_id, movie_id, train_data, test_data, genre_columns, numerical_features)
    
    if rule_rating is None or clustering_rating is None:
        return None
    
    combined_rating = 0.7 * rule_rating + 0.3 * clustering_rating
    denormalized_rating = combined_rating * scaler.scale_[0] + scaler.mean_[0]
    
    denormalized_rating = max(0.5, min(denormalized_rating, 5))
    return round_to_neares_half(denormalized_rating)

# Tune KMeans clustering parameters using GridSearchCV
def tune_kmeans(train_data, genre_columns, numerical_features):
    user_features = train_data[['userId'] + numerical_features + genre_columns].drop_duplicates().drop(columns=['userId'])
    
    param_grid = {'n_clusters': [5, 10, 15, 20]}
    kmeans = KMeans(random_state=42)
    grid_search = GridSearchCV(kmeans, param_grid, cv=3)
    grid_search.fit(user_features)
    
    return grid_search.best_params_['n_clusters']

# Evaluate recommender system performance using MAE and RMSE
def evaluate_recommender(test_data, train_data, scaler, genre_columns, numerical_features):
    actual_ratings = []
    predicted_ratings = []

    best_n_clusters = tune_kmeans(train_data, genre_columns, numerical_features)
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    kmeans.fit(train_data[['userId'] + numerical_features + genre_columns].drop_duplicates().drop(columns=['userId']))
    
    for index, row in test_data.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        
        predicted_rating = combined_rating(user_id, movie_id, train_data, test_data, scaler, genre_columns, numerical_features)
        
        if predicted_rating is not None:
            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_rating)
    
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    return mae, rmse
