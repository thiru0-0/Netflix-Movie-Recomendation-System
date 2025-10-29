import difflib
from typing import List, Tuple

import pandas as pd
import streamlit as st

try:
    # Optional: if scikit-learn is installed, we can provide a TF-IDF fallback
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names we rely on
    expected_cols = {"N_id", "Title", "Recommendations"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # Ensure types and cleaned values
    df["Title"] = df["Title"].astype(str).str.strip()
    df["N_id"] = df["N_id"].astype(str).str.strip()
    df["Recommendations"] = df["Recommendations"].astype(str).fillna("")
    return df


def best_title_match(query: str, titles: List[str]) -> Tuple[str, float]:
    if not query:
        return "", 0.0
    match = difflib.get_close_matches(query, titles, n=1, cutoff=0.0)
    if not match:
        return "", 0.0
    best = match[0]
    ratio = difflib.SequenceMatcher(None, query.lower(), best.lower()).ratio()
    return best, ratio


def parse_recommendation_ids(raw: str) -> List[str]:
    if not raw or not isinstance(raw, str):
        return []
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items


def tfidf_fallback(df: pd.DataFrame, seed_title: str, top_k: int = 10) -> List[str]:
    if not SKLEARN_AVAILABLE:
        return []
    # Build a simple text corpus from available columns
    text_cols = []
    for col in ["Title", "Main Genre", "Sub Genres"]:
        if col in df.columns:
            text_cols.append(col)
    if not text_cols:
        text_cols = ["Title"]
    corpus = (df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)).tolist()
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(corpus)
        # Find index of seed_title
        seed_idx = df.index[df["Title"].str.lower() == seed_title.lower()]
        if len(seed_idx) == 0:
            return []
        seed_idx = seed_idx[0]
        sims = cosine_similarity(X[seed_idx], X).ravel()
        # Exclude self and return top_k similar
        similar_indices = sims.argsort()[::-1]
        out = []
        for idx in similar_indices:
            if idx == seed_idx:
                continue
            out.append(df.iloc[idx]["Title"])
            if len(out) >= top_k:
                break
        return out
    except Exception:
        return []


def recommend_titles(df: pd.DataFrame, seed_title: str, top_k: int = 10) -> List[str]:
    # Get the row for the selected title
    row = df[df["Title"].str.lower() == seed_title.lower()]
    if row.empty:
        return []
    row = row.iloc[0]
    rec_ids = parse_recommendation_ids(row.get("Recommendations", ""))
    if rec_ids:
        id_to_title = {str(r["N_id"]): r["Title"] for _, r in df.iterrows()}
        titles = [id_to_title[i] for i in rec_ids if i in id_to_title]
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for t in titles:
            if t not in seen and t.lower() != seed_title.lower():
                seen.add(t)
                ordered.append(t)
            if len(ordered) >= top_k:
                break
        if ordered:
            return ordered
    # Fallback to TF-IDF similarity when no curated recs found
    return tfidf_fallback(df, seed_title, top_k)


def main() -> None:
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
    st.title("🎬 Movie Recommender")
    st.caption("Type a movie title, get related movies. Uses your Netflix dataset.")

    with st.sidebar:
        st.markdown("**Settings**")
        top_k = st.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=1)
        data_path = st.text_input("CSV path", value="netflix_data.csv")

    try:
        df = load_data(data_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    titles = df["Title"].dropna().astype(str).tolist()

    query = st.text_input("Enter a movie title", placeholder="e.g., The Green Mile")
    if st.button("Recommend") or (query and st.session_state.get("auto_run", False)):
        if not query.strip():
            st.warning("Please enter a movie title.")
        else:
            best, score = best_title_match(query.strip(), titles)
            if not best:
                st.warning("No close match found. Try another title.")
                return

            st.subheader(f"Results for: {best}")
            if score < 0.5:
                st.info("The match is uncertain; results may be less accurate.")

            with st.spinner("Finding recommendations..."):
                recs = recommend_titles(df, best, top_k=top_k)

            if not recs:
                st.warning("No recommendations found.")
            else:
                for i, t in enumerate(recs, 1):
                    st.write(f"{i}. {t}")


if __name__ == "__main__":
    main()


