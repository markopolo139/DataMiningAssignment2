# Data Mining Assignment 2
## MovieLens Latest Datasets 
This dataset, known as ml-latest, documents the activity of users on MovieLens, a movie recommendation platform, by capturing both 5-star ratings and free-text tagging. It encompasses a vast volume of data, comprising 33,832,162 ratings and 2,328,315 tag applications across 86,537 movies. These records were contributed by 330,975 users spanning from January 9, 1995, to July 20, 2023. The dataset itself was assembled on July 20, 2023.

For inclusivity, users were randomly chosen, ensuring that each selected user had rated at least one movie. However, no demographic information is provided. Users are solely identified by an ID without any additional details.

The dataset comprises several files, namely genome-scores.csv, genome-tags.csv, links.csv, movies.csv, ratings.csv, and tags.csv. Further elaboration on the contents and utilization of these files will be provided.

It's crucial to note that this dataset is designated for development purposes, subject to potential modifications over time. Hence, it's not advisable for producing shared research findings. Researchers seeking datasets for benchmarking purposes are advised to explore alternative benchmark datasets.Our task is to utilize the model trained on the training set to forecast whether each passenger in the test set survived the Titanic's tragic sinking or not. This evaluation is crucial as it assesses the real-world applicability and effectiveness of the model

#### Brief description of each data variable:

The ratings data file (ratings.csv) consists of all ratings, organized in the following format: userId, movieId, rating, and timestamp. Each line in this file, excluding the header row, represents a single rating of a movie by a user. The lines are sorted primarily by userId and then by movieId within each user entry.

Ratings are recorded on a scale of 0.5 to 5.0 stars, allowing for half-star increments.

Timestamps in this file represent the number of seconds elapsed since midnight Coordinated Universal Time (UTC) on January 1, 1970.

In the movies data file (movies.csv), movie information is stored. Each line, excluding the header row, corresponds to a single movie entry, containing the movieId, title, and genres. Movie titles may either be manually inputted or imported from themoviedb.org, often including the release year within parentheses. Note that errors and discrepancies might exist in these titles.

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
- **Data Loading**: Loaded the Titanic dataset consisting of passenger information such as survival status, ticket class, gender, age, family relations, ticket details, fare, cabin number, and port of embarkation.

- **Data Preprocessing**:
    - *Handling Missing Values*: Dropped columns with no relevant information and removed rows with missing values, considering the importance of features in the dataset.
    - *Encoding Categorical Variables*: Converted categorical variables like 'Sex' and 'Embarked' into numerical labels for machine learning algorithms to process.
    - *Normalization and Standardization*: We tested both preprocessing techniques and we decided to perform the data scaling using StandardScaler over the MinMaxScaler.

- **Feature Selection**: Recursive Feature Elimination (RFE): Used RFE with Logistic Regression and Random Forest Classifier to select the most important features for prediction.

- **Feature Extraction**: Principal Component Analysis (PCA): Applied PCA to reduce the dimensionality of the dataset while retaining important information, creating new features.

- **Model Training and Evaluation**:
    - Trained machine learning models (Logistic Regression, SVC) on the training dataset with selected and extracted features.
    - Evaluated the models' performance on the validation set using metrics such as accuracy and classification report.
- **Prediction on Test Set**: Utilized the trained models to predict the survival outcomes of passengers in the test set, who did not have their survival status provided.
