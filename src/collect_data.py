import os
import json
import requests
import pandas as pd

# Thu muc project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ca 3 file se duoc luu tai day
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "collected")

FILES = {
    "tmdb_5000_movies.csv":
        "https://raw.githubusercontent.com/masiod/TMDb_movie_data/master/tmdb_5000_movies.csv",
    "tmdb_5000_credits.csv":
        "https://raw.githubusercontent.com/masiod/TMDb_movie_data/master/tmdb_5000_credits.csv"
}


def tai_file(ten_file, url):
    path = os.path.join(OUTPUT_DIR, ten_file)

    # Neu file da ton tai thi tai lai de cap nhat
    print("Dang tai:", ten_file)
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    print("Da luu:", path)


def doc_json(value):
    try:
        return json.loads(value) if isinstance(value, str) else []
    except (json.JSONDecodeError, TypeError):
        return []


def lay_dao_dien(crew):
    data = doc_json(crew)

    for person in data:
        if person.get("job") == "Director":
            return person.get("name", "")

    return ""


def lay_dien_vien(cast):
    data = doc_json(cast)
    names = [person.get("name", "") for person in data[:10]]
    return ", ".join(name for name in names if name)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Tai hai file du lieu goc
    for ten_file, url in FILES.items():
        tai_file(ten_file, url)

    # 2. Doc du lieu
    movies_path = os.path.join(
        OUTPUT_DIR, "tmdb_5000_movies.csv"
    )
    credits_path = os.path.join(
        OUTPUT_DIR, "tmdb_5000_credits.csv"
    )

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # 3. Ghep phim voi thong tin dien vien va ekip
    df = movies.merge(
        credits[["movie_id", "cast", "crew"]],
        left_on="id",
        right_on="movie_id",
        how="left"
    )

    # 4. Tao cac cot theo format
    df["Release_Year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year

    df["Duration_min"] = df["runtime"]
    df["IMDb_Rating"] = df["vote_average"]

    # Them so luot binh chon tu du lieu TMDB
    df["Votes"] = df["vote_count"]

    df["Budget_USD"] = df["budget"]
    df["Revenue_USD"] = df["revenue"]
    df["Actors"] = df["cast"].apply(lay_dien_vien)
    df["Director"] = df["crew"].apply(lay_dao_dien)

    # Giu Genre dang JSON de code clean xu ly
    df["Genre"] = df["genres"]

    # 5. Chon dung cac cot dau vao
    df = df[[
        "title",
        "Release_Year",
        "Genre",
        "Duration_min",
        "IMDb_Rating",
        "Votes",
        "Budget_USD",
        "Revenue_USD",
        "Director",
        "Actors"
    ]]

    df = df.rename(columns={"title": "Title"})

    # 6. Luu file trung gian cung thu muc voi 2 file goc
    output_path = os.path.join(
        OUTPUT_DIR, "resultsRaw_completed.csv"
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nDa hoan thanh!")
    print("So phim:", len(df))
    print("Cac cot:", list(df.columns))
    print("Thu muc luu file:", OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nCo loi xay ra:", e)
