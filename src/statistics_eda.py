"""Phần II: Thống kê mô tả và tìm phim có giá trị cực trị."""

import numpy as np
import pandas as pd

# Các cột số có ý nghĩa phân tích (tên chuẩn từ dữ liệu thô).
STAT_COLUMNS = ["Duration_min", "IMDb_Rating", "Budget_USD", "Revenue_USD"]


def calculate_basic_stats(df):
    """Tính trung bình, trung vị và độ lệch chuẩn của các cột số.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim đã làm sạch.

    Trả về khi hoàn thiện:
        tuple[pandas.DataFrame, pandas.DataFrame]: (overall, by_genre).
        overall: mỗi hàng là một cột số; các cột là Mean, Median, Std;
        chỉ mục mang tên column.
        by_genre: thống kê theo từng thể loại với các cột phẳng, ví dụ
        Budget_mean, Budget_median, Budget_std. Cột Movie_Count là số
        phim thuộc thể loại đó; phim đa thể loại được tính vào từng thể
        loại của nó.

    Công việc cần triển khai:
        - Chọn cột số có ý nghĩa phân tích, không dùng mã định danh phim.
        - Tính Mean, Median và Std mẫu (ddof=1), bỏ qua giá trị thiếu.
        - Giữ giá trị thiếu nếu không đủ quan sát để tính thống kê.
        - Xử lý đầu vào rỗng hoặc không có cột số một cách nhất quán.
        - Trả về bảng; tầng gọi lưu data/results2.csv với index=True
          để giữ tên các cột được thống kê.
    """
    cols = [c for c in STAT_COLUMNS if df is not None and c in df.columns]

    # Đầu vào rỗng hoặc không có cột số -> hai bảng rỗng
    if df is None or df.empty or not cols:
        empty = pd.DataFrame(columns=["Mean", "Median", "Std"])
        empty.index.name = "column"
        return empty, pd.DataFrame()

    # Ép kiểu số trên bản sao: "N/a" -> NaN, inf -> NaN
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    # 1. Thống kê toàn bộ dữ liệu
    overall = pd.DataFrame({
        "Mean": numeric.mean(),
        "Median": numeric.median(),
        "Std": numeric.std(ddof=1),  # NaN nếu chỉ có 1 quan sát
    })
    overall.index.name = "column"

    # 2. Thống kê theo thể loại (phim đa thể loại tách bằng ",")
    if "Genre" not in df.columns:
        return overall, pd.DataFrame()

    genres = (
        df["Genre"].fillna("Unknown").astype(str).str.split(",")
        .explode().str.strip().rename("Genre")
    )
    # join theo chỉ mục: một phim nhiều thể loại -> nhiều dòng
    expanded = numeric.join(genres, how="inner")
    grouped = expanded.groupby("Genre")
    by_genre = grouped[cols].agg(["mean", "median", "std"])

    # Gộp hai tầng tên cột để bảng xuất CSV/Excel chỉ có một hàng tiêu đề,
    # ví dụ: Budget_mean, Budget_median, Budget_std.
    by_genre.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in by_genre.columns.to_flat_index()
    ]
    by_genre.insert(
        0,
        "Movie_Count",
        grouped.size().reindex(by_genre.index).astype("int64"),
    )

    return overall, by_genre


def get_top_bottom_movies(df, column, top_n=3):
    """Tìm tất cả phim có giá trị lớn nhất và nhỏ nhất trong một cột.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim đã làm sạch.
        column (str): Tên cột số cần tìm cực trị.
        top_n (int): Số phim lấy ở mỗi đầu bảng, mặc định là 3.

    Trả về khi hoàn thiện:
        tuple[pandas.DataFrame, pandas.DataFrame]: Hai bảng gồm top_n phim
        theo thứ tự (điểm cao nhất, điểm thấp nhất), có cột Title và column.

    Công việc cần triển khai:
        - Kiểm tra cột tồn tại và có kiểu số; bỏ qua giá trị thiếu.
        - Dùng .nlargest(top_n, column) và .nsmallest(top_n, column).
        - Nếu không có giá trị hợp lệ, trả về hai bảng rỗng cùng cấu trúc.
        - Không sửa đổi DataFrame đầu vào.
    """
    if "Title" not in df.columns or column not in df.columns:
        raise ValueError(f"Thiếu cột 'Title' hoặc '{column}' trong dữ liệu.")

    values = pd.to_numeric(df[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)

    data = pd.DataFrame({"Title": df["Title"], column: values})
    data = data.dropna(subset=[column])  # bỏ giá trị thiếu

    if data.empty:  # không có giá trị hợp lệ
        empty = pd.DataFrame(columns=["Title", column])
        return empty, empty.copy()

    top_movies = data.nlargest(top_n, column)
    bottom_movies = data.nsmallest(top_n, column)
    return top_movies, bottom_movies
