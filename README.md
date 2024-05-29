# Data Mining Assignment 2
## MovieLens Latest Datasets 
This dataset, known as ml-latest, documents the activity of users on MovieLens, a movie recommendation platform, by capturing both 5-star ratings and free-text tagging. It encompasses a vast volume of data, comprising 33,832,162 ratings and 2,328,315 tag applications across 86,537 movies. These records were contributed by 330,975 users spanning from January 9, 1995, to July 20, 2023. The dataset itself was assembled on July 20, 2023.

For inclusivity, users were randomly chosen, ensuring that each selected user had rated at least one movie. However, no demographic information is provided. Users are solely identified by an ID without any additional details.

The dataset comprises several files, namely genome-scores.csv, genome-tags.csv, links.csv, movies.csv, ratings.csv, and tags.csv. Further elaboration on the contents and utilization of these files will be provided.

It's crucial to note that this dataset is designated for development purposes, subject to potential modifications over time. Hence, it's not advisable for producing shared research findings. Researchers seeking datasets for benchmarking purposes are advised to explore alternative benchmark datasets.Our task is to utilize the model trained on the training set to forecast whether each passenger in the test set survived the Titanic's tragic sinking or not. This evaluation is crucial as it assesses the real-world applicability and effectiveness of the model

#### Brief description of each data variable:

The ratings data file **(ratings.csv)** consists of all ratings, organized in the following format: userId, movieId, rating, and timestamp. Each line in this file, excluding the header row, represents a single rating of a movie by a user. The lines are sorted primarily by userId and then by movieId within each user entry.

Ratings are recorded on a scale of 0.5 to 5.0 stars, allowing for half-star increments.

Timestamps in this file represent the number of seconds elapsed since midnight Coordinated Universal Time (UTC) on January 1, 1970.

In the movies data file **(movies.csv)**, movie information is stored. Each line, excluding the header row, corresponds to a single movie entry, containing the movieId, title, and genres. Movie titles may either be manually inputted or imported from themoviedb.org, often including the release year within parentheses. Note that errors and discrepancies might exist in these titles.

Genres are represented as a list separated by pipes ('|') and are selected from the following categories:
  - Action
  - Adventure
  - Animation
  - Children's
  - Comedy
  - Crime
  - Documentary
  - Drama
  - Fantasy
  - Film-Noir
  - Horror
  - Musical
  - Mystery
  - Romance
  - Sci-Fi
  - Thriller
  - War
  - Western
  - (no genres listed)

The structure of the tags data file **(tags.csv)** is organized as follows:

- Each line, excluding the header row, represents a single tag applied by a user to a movie.
- The format of each line consists of four elements: userId, movieId, the tag itself, and a timestamp.
- Entries in the file are sorted primarily by userId, then within each user, by movieId.

Tags serve as user-generated metadata for movies, providing additional information or descriptors beyond basic categorization. Typically, each tag is a single word or a short phrase. The significance and purpose of a specific tag vary according to the user who applies it, reflecting their individual interpretation or perception of the movie.

## Libraries Used:
- **pandas**: For data manipulation and analysis. It provides data structures and functions to work with structured data, such as data frames.
- **numpy**: Numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.
- **copy**: Module provides functions for creating shallow and deep copies of objects in Python.
- **scikit-learn (sklearn)**:
    - **StandardScaler**: Used for standardizing features by removing the mean and scaling to unit variance.
    - **MinMaxScaler**: Used for scaling features to a specified range (usually between 0 and 1).
    - **train_test_split**: Splits arrays or matrices into random train and test subsets.


## Installation of Libraries
**Pip freeze**: You can install the dependencies by running the following command in your terminal.

$ pip install -r Requirements.txt 

or 

$ pip freeze > Requirements.txt to get the requirements and later above command

## Brief descrition of what was done in the project
### 1. Data Loading and Preprocessing

  1.1. Load the movie, ratings, tags, and links data.
  
  1.2. Print the total number of movies and unique users.
  
  1.3. Drop unnecessary columns like timestamps.
  
  1.4. Merge ratings and movies data.
  
  1.5. Merge tags data with ratings and movies.
  
  1.6. Drop rows with missing values and duplicate entries.

### 2. Feature Engineering
  2.1. Calculate average rating for each movie.
  
  2.2. Transform the 'genres' attribute into multiple binary attributes.
  
  2.3. Drop redundant columns like 'genres', 'title', and '(no genres listed)'.
  
  2.4. Compute the total count of genres per movie.
  
  2.5. Fill missing values with zero.
  
  2.6. Scale numerical features ('rating', 'average_rating', 'total_genres') using StandardScaler.

### 3. Rating Prediction Functions
  3.1. Implement a rule-based rating prediction function based on user and movie genre preferences.
  
  3.2. Implement a clustering-based rating prediction function using KMeans clustering.
  
  3.3. Define a combined rating prediction function that averages ratings from the rule-based and clustering-based methods.

### 4. Evaluation
  4.1. Define a function to evaluate the model.
  
  4.2. Split the data into test and train sets.
  
  4.3. Predict ratings for test data using the combined rating function.
  
  4.4. Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for model evaluation.

### 5. Results
  
  5.1. Print the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of the model.
