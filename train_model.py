import pandas as pd
import numpy as np
import ast
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading datasets...")

movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

print("Movies:", movies.shape)
print("Credits:", credits.shape)


# ==========================================
# 2. MERGE DATASETS
# ==========================================

credits = credits.rename(columns={"movie_id": "id"})

movies = movies.merge(
    credits[["id", "cast", "crew"]],
    on="id",
    how="left"
)

print("Merged dataset:", movies.shape)


# ==========================================
# 3. KEEP IMPORTANT COLUMNS
# ==========================================

movies = movies[
    [
        "id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
        "budget",
        "revenue",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count",
        "release_date"
    ]
].copy()


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

movies["overview"] = movies["overview"].fillna("")
movies["genres"] = movies["genres"].fillna("[]")
movies["keywords"] = movies["keywords"].fillna("[]")
movies["cast"] = movies["cast"].fillna("[]")
movies["crew"] = movies["crew"].fillna("[]")

movies["runtime"] = movies["runtime"].fillna(
    movies["runtime"].median()
)

movies["budget"] = movies["budget"].fillna(0)
movies["revenue"] = movies["revenue"].fillna(0)
movies["popularity"] = movies["popularity"].fillna(0)
movies["vote_average"] = movies["vote_average"].fillna(0)
movies["vote_count"] = movies["vote_count"].fillna(0)

movies["release_date"] = movies["release_date"].fillna("")


# ==========================================
# 5. EXTRACT JSON DATA
# ==========================================

def extract_names(text):
    try:
        data = ast.literal_eval(text)

        if isinstance(data, list):
            return [item["name"] for item in data if "name" in item]

        return []

    except:
        return []


def extract_cast(text):
    try:
        data = ast.literal_eval(text)

        if isinstance(data, list):
            return [
                item["name"]
                for item in data[:5]
                if "name" in item
            ]

        return []

    except:
        return []


def extract_director(text):
    try:
        data = ast.literal_eval(text)

        if isinstance(data, list):
            for item in data:
                if item.get("job") == "Director":
                    return item.get("name", "")

        return ""

    except:
        return ""


movies["genres_list"] = movies["genres"].apply(extract_names)
movies["keywords_list"] = movies["keywords"].apply(extract_names)
movies["cast_list"] = movies["cast"].apply(extract_cast)
movies["director"] = movies["crew"].apply(extract_director)


# ==========================================
# 6. CREATE TEXT FEATURES
# ==========================================

movies["genres_text"] = movies["genres_list"].apply(
    lambda x: " ".join(x)
)

movies["keywords_text"] = movies["keywords_list"].apply(
    lambda x: " ".join(x)
)

movies["cast_text"] = movies["cast_list"].apply(
    lambda x: " ".join(x)
)


movies["combined_features"] = (
    movies["overview"]
    + " "
    + movies["genres_text"]
    + " "
    + movies["keywords_text"]
    + " "
    + movies["cast_text"]
    + " "
    + movies["director"]
)


# ==========================================
# 7. TF-IDF
# ==========================================

print("\nCreating TF-IDF vectors...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = tfidf.fit_transform(
    movies["combined_features"]
)

print("TF-IDF matrix:", tfidf_matrix.shape)


# ==========================================
# 8. COSINE SIMILARITY
# ==========================================

print("\nCalculating cosine similarity...")

similarity_matrix = cosine_similarity(tfidf_matrix)

print("Similarity matrix:", similarity_matrix.shape)


# ==========================================
# 9. RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

movies["release_year"] = pd.to_datetime(
    movies["release_date"],
    errors="coerce"
).dt.year

movies["release_year"] = movies["release_year"].fillna(0)


features = [
    "budget",
    "revenue",
    "runtime",
    "popularity",
    "vote_count",
    "release_year"
]

X = movies[features]
y = movies["vote_average"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


random_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)


# ==========================================
# 10. MODEL EVALUATION
# ==========================================

predictions = random_forest.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nRandom Forest Evaluation")
print("------------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# 11. SAVE MODELS
# ==========================================

print("\nSaving models...")

joblib.dump(
    tfidf,
    "model/tfidf.pkl"
)

joblib.dump(
    similarity_matrix,
    "model/similarity.pkl"
)

joblib.dump(
    random_forest,
    "model/random_forest.pkl"
)

# Save processed movie data
movies.to_pickle(
    "model/movies.pkl"
)


# ==========================================
# 12. FINISHED
# ==========================================

print("\n================================")
print("MovieWorld ML training complete!")
print("================================")

print("\nSaved files:")

print("✓ model/tfidf.pkl")
print("✓ model/similarity.pkl")
print("✓ model/random_forest.pkl")
print("✓ model/movies.pkl")