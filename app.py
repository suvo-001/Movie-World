import streamlit as st
import pandas as pd

from recommender import (
    recommend_movies,
    get_poster
)


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MovieWorld",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# CSS
# ==========================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(90, 70, 180, 0.20),
                transparent 32%
            ),
            radial-gradient(
                circle at 85% 15%,
                rgba(0, 180, 255, 0.14),
                transparent 32%
            ),
            #070910;
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .logo {
        text-align: center;
        font-size: 58px;
        font-weight: 900;
        letter-spacing: -3px;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #9da6ba;
        font-size: 18px;
        margin-top: 5px;
        margin-bottom: 45px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 18px;
    }

    .selected-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .selected-meta {
        color: #aeb7ca;
        font-size: 15px;
        margin-bottom: 15px;
        line-height: 1.8;
    }

    .overview {
        color: #c4cada;
        line-height: 1.7;
        font-size: 15px;
    }

    .movie-title {
        font-size: 18px;
        font-weight: 750;
        margin-top: 10px;
        line-height: 1.3;
    }

    .movie-year {
        color: #8f98ad;
        font-size: 13px;
        margin-top: 4px;
    }

    .rating {
        margin-top: 7px;
        font-size: 14px;
    }

    .info-box {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 18px;
        margin-top: 10px;
        color: #c4cada;
        line-height: 1.7;
    }

    .detail-meta {
        color: #aeb7ca;
        font-size: 15px;
        line-height: 2;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="logo">🎬 MovieWorld</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Movie Recommendation System'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# LOAD MOVIES
# ==========================================

movies = pd.read_pickle(
    "model/movies.pkl"
)

movie_titles = sorted(
    movies["title"]
    .dropna()
    .unique()
    .tolist()
)


# ==========================================
# MOVIE SEARCH
# ==========================================

st.markdown(
    '<div class="section-title">'
    '🔎 Find Your Next Movie'
    '</div>',
    unsafe_allow_html=True
)

selected_movie = st.selectbox(
    "Search or select a movie",
    movie_titles,
    index=None,
    placeholder="Type a movie name..."
)


# ==========================================
# SHOW SELECTED MOVIE
# ==========================================

if selected_movie:

    selected_rows = movies[
        movies["title"] == selected_movie
    ]

    if not selected_rows.empty:

        selected = selected_rows.iloc[0]

        year = selected["release_year"]

        if year == 0 or pd.isna(year):
            year = "N/A"
        else:
            year = int(year)

        poster = get_poster(
            selected_movie,
            year
        )

        st.markdown(
            '<div class="section-title">'
            '🎬 Selected Movie'
            '</div>',
            unsafe_allow_html=True
        )

        left, right = st.columns(
            [1, 2],
            gap="large"
        )

        with left:

            if poster:

                st.image(
                    poster,
                    use_container_width=True
                )

            else:

                st.info(
                    "Poster unavailable"
                )

        with right:

            st.markdown(
                f'<div class="selected-title">'
                f'{selected_movie}'
                f'</div>',
                unsafe_allow_html=True
            )

            genres = selected["genres_list"]

            if isinstance(genres, list):
                genre_text = " • ".join(genres)
            else:
                genre_text = "N/A"

            runtime = selected["runtime"]

            if pd.isna(runtime) or runtime == 0:
                runtime_text = "N/A"
            else:
                runtime_text = f"{int(runtime)} min"

            st.markdown(
                f'<div class="selected-meta">'
                f'📅 {year} &nbsp;&nbsp; '
                f'⏱️ {runtime_text} &nbsp;&nbsp; '
                f'⭐ {selected["vote_average"]:.1f}'
                f'<br><br>'
                f'🏷️ {genre_text}'
                f'</div>',
                unsafe_allow_html=True
            )

            overview = selected["overview"]

            if not overview:
                overview = (
                    "No overview available for this movie."
                )

            st.markdown(
                f'<div class="overview">'
                f'{overview}'
                f'</div>',
                unsafe_allow_html=True
            )


# ==========================================
# RECOMMEND BUTTON
# ==========================================

if st.button(
    "✨ Get Recommendations",
    use_container_width=True
):

    if not selected_movie:

        st.warning(
            "Please select a movie first."
        )

    else:

        with st.spinner(
            "Finding movies you may like..."
        ):

            recommendations = recommend_movies(
                selected_movie,
                10
            )

        if recommendations is None:

            st.error(
                "Movie not found."
            )

        else:

            st.session_state[
                "recommendations"
            ] = recommendations

            st.session_state[
                "selected_movie"
            ] = selected_movie


# ==========================================
# RECOMMENDATIONS
# ==========================================

if "recommendations" in st.session_state:

    recommendations = st.session_state[
        "recommendations"
    ]

    st.markdown(
        '<div class="section-title">'
        '✨ Recommended For You'
        '</div>',
        unsafe_allow_html=True
    )

    columns = st.columns(5)

    for i, movie in enumerate(
        recommendations
    ):

        with columns[i % 5]:

            # ==================================
            # POSTER
            # ==================================

            if movie["poster"]:

                st.image(
                    movie["poster"],
                    use_container_width=True
                )

            else:

                st.markdown(
                    """
                    <div style="
                        height:300px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:#151925;
                        border-radius:14px;
                        color:#7f879a;
                        text-align:center;
                    ">
                    🎬<br>
                    Poster unavailable
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ==================================
            # MOVIE TITLE
            # ==================================

            st.markdown(
                f'<div class="movie-title">'
                f'{movie["title"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            # ==================================
            # YEAR
            # ==================================

            st.markdown(
                f'<div class="movie-year">'
                f'📅 {movie["release_year"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            # ==================================
            # RATING
            # ==================================

            st.markdown(
                f'<div class="rating">'
                f'⭐ {movie["rating"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            # ==================================
            # DETAILS BUTTON
            # ==================================

            if st.button(
                "View Details",
                key=f"details_{i}",
                use_container_width=True
            ):

                st.session_state[
                    "selected_details"
                ] = movie


# ==========================================
# MOVIE DETAILS
# ==========================================

if "selected_details" in st.session_state:

    movie = st.session_state[
        "selected_details"
    ]

    st.markdown(
        '<div class="section-title">'
        '🎥 Movie Details'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1, 2],
        gap="large"
    )

    # ======================================
    # POSTER
    # ======================================

    with left:

        if movie["poster"]:

            st.image(
                movie["poster"],
                use_container_width=True
            )

        else:

            st.info(
                "Poster unavailable"
            )

    # ======================================
    # DETAILS
    # ======================================

    with right:

        st.markdown(
            f'<div class="selected-title">'
            f'{movie["title"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        genres = movie["genres"]

        if isinstance(genres, list):
            genre_text = " • ".join(genres)
        else:
            genre_text = "N/A"

        # Runtime
        runtime = movie.get("runtime", None)

        if runtime is None or pd.isna(runtime) or runtime == 0:
            runtime_text = "N/A"
        else:
            runtime_text = f"{int(runtime)} min"

        st.markdown(
            f'<div class="detail-meta">'
            f'🏷️ <b>Genre:</b> {genre_text}<br>'
            f'📅 <b>Release Year:</b> {movie["release_year"]}<br>'
            f'⏱️ <b>Runtime:</b> {runtime_text}<br>'
            f'⭐ <b>Rating:</b> {movie["rating"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        overview = movie["overview"]

        if not overview:
            overview = (
                "No details available for this movie."
            )

        st.markdown(
            f'<div class="info-box">'
            f'<b>📝 Details</b>'
            f'<br><br>'
            f'{overview}'
            f'</div>',
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#687186;
        margin-top:70px;
        font-size:13px;
    ">
        MovieWorld • AI-Powered Movie Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)