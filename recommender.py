import joblib
import pandas as pd
import requests
import os

from dotenv import load_dotenv


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")


# ==========================================
# LOAD ML FILES
# ==========================================

movies = pd.read_pickle(
    "model/movies.pkl"
)

similarity_matrix = joblib.load(
    "model/similarity.pkl"
)


# ==========================================
# GET POSTER FROM OMDB
# ==========================================

def get_poster(title, year):

    if not OMDB_API_KEY:
        return None

    try:

        params = {
            "apikey": OMDB_API_KEY,
            "t": title,
            "type": "movie"
        }

        if year != "N/A":
            params["y"] = int(year)

        response = requests.get(
            "https://www.omdbapi.com/",
            params=params,
            timeout=10
        )

        data = response.json()

        poster = data.get("Poster")

        if poster and poster != "N/A":
            return poster

        return None

    except Exception:
        return None


# ==========================================
# RECOMMEND MOVIES
# ==========================================

def recommend_movies(
    movie_title,
    number_of_recommendations=10
):

    # Clean movie title
    movie_title = movie_title.strip().lower()


    # ======================================
    # FIND SELECTED MOVIE
    # ======================================

    matches = movies[
        movies["title"].str.lower() == movie_title
    ]

    if matches.empty:
        return None


    # Get original dataset index
    movie_index = matches.index[0]


    # ======================================
    # COSINE SIMILARITY
    # ======================================

    similarity_scores = list(
        enumerate(
            similarity_matrix[movie_index]
        )
    )


    # Sort from most similar to least similar
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )


    # ======================================
    # CREATE RECOMMENDATIONS
    # ======================================

    recommendations = []


    for index, similarity_score in similarity_scores[1:]:

        movie = movies.iloc[index]


        # ==================================
        # RELEASE YEAR
        # ==================================

        release_year = movie["release_year"]

        if (
            release_year == 0
            or pd.isna(release_year)
        ):
            release_year = "N/A"

        else:
            release_year = int(
                release_year
            )


        # ==================================
        # RUNTIME
        # ==================================

        runtime = movie["runtime"]

        if (
            runtime == 0
            or pd.isna(runtime)
        ):
            runtime = "N/A"

        else:
            runtime = int(runtime)


        # ==================================
        # GET REAL POSTER
        # ==================================

        poster = get_poster(
            movie["title"],
            release_year
        )


        # ==================================
        # GENRES
        # ==================================

        genres = movie["genres_list"]

        if not isinstance(genres, list):
            genres = []


        # ==================================
        # OVERVIEW
        # ==================================

        overview = movie["overview"]

        if pd.isna(overview) or not overview:
            overview = (
                "No details available "
                "for this movie."
            )


        # ==================================
        # ADD MOVIE
        # ==================================

        recommendations.append({

            "title":
                movie["title"],

            "rating":
                round(
                    float(
                        movie["vote_average"]
                    ),
                    2
                ),

            "genres":
                genres,

            "overview":
                overview,

            "release_year":
                release_year,

            "runtime":
                runtime,

            "poster":
                poster

        })


        # ==================================
        # STOP AFTER REQUIRED NUMBER
        # ==================================

        if (
            len(recommendations)
            >= number_of_recommendations
        ):
            break


    return recommendations


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    result = recommend_movies(
        "Inception",
        5
    )


    if result is None:

        print(
            "Movie not found."
        )


    else:

        print(
            "\nMovieWorld Recommendations"
        )

        print(
            "=========================="
        )


        for movie in result:

            print(
                f"\n{movie['title']}"
            )

            print(
                f"Rating: "
                f"{movie['rating']}"
            )

            print(
                f"Genre: "
                f"{', '.join(movie['genres'])}"
            )

            print(
                f"Year: "
                f"{movie['release_year']}"
            )

            print(
                f"Runtime: "
                f"{movie['runtime']} min"
            )

            print(
                f"Poster: "
                f"{movie['poster']}"
            )

            print(
                f"Details: "
                f"{movie['overview']}"
            )