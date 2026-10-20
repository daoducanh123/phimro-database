"""Phần I: Đọc, chuẩn hóa và làm sạch dữ liệu phim IMDb/TMDB.

Hàm load_and_clean_data do Tổ 1 cài đặt; đây là bản gốc (single source of
truth) mà src/data_clean.py và app.py đều tham chiếu.
"""

import json
import pathlib
import re

import numpy as np
import pandas as pd

# 9 cột chuẩn đầu ra theo README mục 5.
_TARGET_COLUMNS = [
    "Title",
    "Year",
    "Genre",
    "Duration",
    "Rating",
    "Budget",
    "Revenue",
    "Director",
    "Cast",
]

# Ánh xạ cột dữ liệu thô (collect_data.py) về tên rút gọn dùng nội bộ.
_RENAME_MAP = {
    "Release_Year": "Year",
    "IMDb_Rating": "Rating",
    "Budget_USD": "Budget",
    "Revenue_USD": "Revenue",
    "Duration_min": "Duration",
    "Actors": "Cast",
}

# Trong dữ liệu TMDB, 0 ở các cột này biểu thị "chưa công bố/không có dữ liệu",
# không phải một ngân sách, doanh thu, thời lượng hay điểm đánh giá có ý nghĩa.
_NUMERIC_COLUMNS = ("Year", "Duration", "Rating", "Budget", "Revenue")


def _clean_genre(val):
    """Trích tên thể loại từ chuỗi JSON cột Genre (có fallback regex)."""
    if pd.isna(val):
        return np.nan
    try:
        text = str(val).strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        text = text.replace('""', '"')
        data = json.loads(text)
        names = [item["name"] for item in data if "name" in item]
        return ", ".join(names) if names else np.nan
    except Exception:
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', str(val))
        return ", ".join(names) if names else str(val)


def _clean_currency(val):
    """Ép chuỗi tiền tệ ('$1,234', 'N/a', rỗng) về float, lỗi thành NaN."""
    if pd.isna(val) or str(val).strip() in ["", "0", "N/a", "nan", "None"]:
        return np.nan
    clean_val = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(clean_val)
    except ValueError:
        return np.nan


def load_and_clean_data(raw_file_path):
    """Đọc dữ liệu thô và trả về DataFrame đã được làm sạch.

    Tham số:
        raw_file_path (str hoặc pathlib.Path): Đường dẫn tệp dữ liệu thô.

    Trả về:
        pandas.DataFrame: Dữ liệu 9 cột chuẩn (Title, Year, Genre, Duration,
        Rating, Budget, Revenue, Director, Cast); cột số đã ép float64.

    Các bước:
        1. Đọc CSV bằng pandas.
        2. Đổi tên cột thô về tên rút gọn chuẩn.
        3. Parse Genre từ chuỗi JSON thành "Hành động, Phiêu lưu".
        4. Ép cột số về float; 0/chuỗi lỗi thành NaN theo quy ước TMDB.
        5. Loại trùng lặp theo (Title, Year); sắp xếp Title A-Z, Year giảm dần.
        6. Chỉ giữ 9 cột chuẩn có mặt trong dữ liệu.
        7. Xuất data/cleaned/results.csv (utf-8-sig) và trả về DataFrame.

    Ném:
        FileNotFoundError: khi đường dẫn tệp không tồn tại.
    """
    path = pathlib.Path(raw_file_path)

    # 1. Đọc file dữ liệu thô.
    df = pd.read_csv(path)

    # 2. Đổi tên cột về chuẩn.
    df = df.rename(columns=_RENAME_MAP)

    # 3. Trích xuất thể loại từ chuỗi JSON.
    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].apply(_clean_genre)

    # 4. Ép Budget/Revenue về float. _clean_currency cũng đổi 0 thành NaN.
    for col in ("Budget", "Revenue"):
        if col in df.columns:
            df[col] = df[col].apply(_clean_currency)

    # TMDB dùng 0 khi chưa có thông tin runtime/vote_average (và đôi khi
    # năm phát hành). Chuẩn hóa mọi cột số về float và coi 0 là thiếu dữ liệu.
    # Dùng tên cột *sau* khi _RENAME_MAP đã được áp dụng; không dùng các tên
    # gốc như Budget_USD hoặc Duration_min ở bước này.
    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)

    # 5. Loại trùng lặp theo (Title, Year) rồi sắp xếp.
    if {"Title", "Year"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["Title", "Year"])
        df = df.sort_values(by=["Title", "Year"], ascending=[True, False])
    elif "Title" in df.columns:
        df = df.sort_values(by="Title", ascending=True)

    # 6. Giữ đúng 9 cột chuẩn có mặt trong dữ liệu.
    existing_cols = [c for c in _TARGET_COLUMNS if c in df.columns]
    df = df[existing_cols]

    # 7. Xuất file sạch để các phần sau đọc (không điền 'N/a' vào cột số).
    df.to_csv("data/cleaned/results.csv", index=False, encoding="utf-8-sig")

    return df
