"""Phần II và III: Tạo biểu đồ Matplotlib để hiển thị trên Streamlit."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler

# Thuộc tính so sánh của biểu đồ radar (README mục 5 - Tổ 3): tên chuẩn ->
# các tên cột chấp nhận được trong dữ liệu.
_RADAR_ATTRIBUTES = {
    "IMDb_Rating": ("IMDb_Rating", "Rating"),
    "Votes": ("Votes",),
    "Budget_USD": ("Budget_USD", "Budget"),
    "Revenue_USD": ("Revenue_USD", "Revenue"),
    "Duration_min": ("Duration_min", "Duration"),
}

# Nhãn hướng đến người xem báo cáo; tên cột kỹ thuật vẫn chỉ dùng nội bộ.
_RADAR_LABELS = {
    "IMDb_Rating": "Điểm TMDB (thang 10)",
    "Votes": "Lượt đánh giá (lượt)",
    "Budget_USD": "Kinh phí (USD)",
    "Revenue_USD": "Doanh thu (USD)",
    "Duration_min": "Thời lượng (phút)",
}

_HISTOGRAM_LABELS = {
    "Duration": ("Thời lượng", "phút"),
    "Duration_min": ("Thời lượng", "phút"),
    "Rating": ("Điểm TMDB", "trên 10"),
    "IMDb_Rating": ("Điểm TMDB", "trên 10"),
    "Budget": ("Kinh phí", "USD"),
    "Budget_USD": ("Kinh phí", "USD"),
    "Revenue": ("Doanh thu", "USD"),
    "Revenue_USD": ("Doanh thu", "USD"),
}
_CURRENCY_COLUMNS = {"Budget", "Budget_USD", "Revenue", "Revenue_USD"}


def _format_histogram_value(value, column):
    """Định dạng trung vị histogram với đơn vị dễ đọc."""
    if column in _CURRENCY_COLUMNS:
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}".replace(".", ",") + " tỷ USD"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.0f} triệu USD"
        return f"{value:,.0f} USD"
    if column in {"Duration", "Duration_min"}:
        return f"{value:.0f} phút"
    if column in {"Rating", "IMDb_Rating"}:
        return f"{value:.1f}".replace(".", ",") + " / 10"
    return f"{value:,.2f}"


def plot_histogram(df, column):
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Cột '{column}' không tồn tại hoặc không phải dữ liệu số.")
        
    data_to_plot = df[column].dropna()
    data_to_plot = data_to_plot[~data_to_plot.isin([np.inf, -np.inf])]
    
    if data_to_plot.empty:
        raise ValueError(f"Không có dữ liệu hợp lệ để vẽ biểu đồ cho cột '{column}'.")

    label, unit = _HISTOGRAM_LABELS.get(column, (column, ""))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=data_to_plot,
        bins=20,
        kde=True,
        color="steelblue",
        edgecolor="white",
        linewidth=0.7,
        ax=ax,
    )

    median = float(data_to_plot.median())
    ax.axvline(
        median,
        color="darkorange",
        linewidth=2,
        label=f"Trung vị: {_format_histogram_value(median, column)}",
    )
    ax.set_title(f"Phân bố {label}", fontsize=14, fontweight="semibold")
    ax.text(
        0.5,
        1.01,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )
    # Với tiền tệ, dùng triệu/tỷ USD trên tick để tránh ký hiệu khoa học.
    max_abs = float(data_to_plot.abs().max())
    if column in _CURRENCY_COLUMNS and max_abs >= 1e9:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _p: f"{v / 1e9:,.1f} tỷ")
        )
    elif column in _CURRENCY_COLUMNS and max_abs >= 1e6:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _p: f"{v / 1e6:,.0f} triệu")
        )
    ax.set_xlabel(f"{label} ({unit})" if unit else label, fontsize=11)
    ax.set_ylabel("Số phim", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    return fig


def plot_radar_chart(df, movie1_name, movie2_name):
    """So sánh đặc trưng của hai phim trên biểu đồ radar.

    Tham số:
        df (pandas.DataFrame): Dữ liệu có tên phim và các đặc trưng số.
        movie1_name (str): Tên phim thứ nhất.
        movie2_name (str): Tên phim thứ hai.

    Trả về khi hoàn thiện:
        matplotlib.figure.Figure: Biểu đồ radar so sánh hai phim.

    Công việc cần triển khai:
        - Thống nhất cột tên phim; báo lỗi nếu không tìm thấy hoặc trùng
          tên nhiều bản ghi, yêu cầu phân biệt bằng năm phát hành/mã phim.
        - Chọn ít nhất ba đặc trưng số có ý nghĩa; xử lý giá trị thiếu.
        - Chuẩn hóa về cùng thang đo dựa trên tập dữ liệu tham chiếu,
          không chỉ hai phim; xử lý riêng đặc trưng có giá trị hằng.
        - Tạo trục cực, khép kín các đa giác, thêm nhãn và chú giải;
          ghi rõ các giá trị hiển thị đã được chuẩn hóa.
        - Trả về Figure, không gọi plt.show hoặc sửa dữ liệu đầu vào;
          tầng gọi đóng hình sau khi hiển thị.

    Ghi chú cài đặt (Tổ 3):
        - Cột tên phim là Title; so khớp chính xác (đã bỏ khoảng trắng thừa).
          Ném ValueError nếu không tìm thấy hoặc trùng nhiều bản ghi.
        - Thuộc tính: IMDb_Rating, Votes, Budget_USD, Revenue_USD,
          Duration_min (chấp nhận tên rút gọn Rating, Budget, Revenue,
          Duration). Thuộc tính thiếu cột hoặc thiếu giá trị ở một trong hai
          phim bị bỏ qua; cần còn ít nhất 3 thuộc tính.
        - MinMaxScaler được học trên toàn bộ df (tập tham chiếu), không chỉ
          hai phim; thuộc tính hằng số được đặt ở mức 0.5.
        - Các giá trị trên biểu đồ đã chuẩn hóa về [0, 1], không phải giá trị
          gốc.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df phải là pandas.DataFrame.")
    if "Title" not in df.columns:
        raise ValueError("Thiếu cột 'Title' để tìm tên phim.")

    # 1. Lọc dòng dữ liệu của từng phim (đúng thứ tự phim 1, phim 2)
    titles = df["Title"].astype(str).str.strip()
    rows = []
    for name in (movie1_name, movie2_name):
        matched = df[titles == str(name).strip()]
        if len(matched) == 0:
            raise ValueError(f"Không tìm thấy phim: '{name}'.")
        if len(matched) > 1:
            raise ValueError(
                f"Có {len(matched)} bản ghi trùng tên '{name}'; "
                "hãy phân biệt bằng năm phát hành hoặc mã phim."
            )
        rows.append(matched.index[0])

    # 2. Chọn thuộc tính so sánh (cột số hợp lệ, tập tham chiếu = toàn bộ df)
    reference = {}
    for canonical, candidates in _RADAR_ATTRIBUTES.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is not None:
            reference[canonical] = (
                pd.to_numeric(df[found], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
    reference = pd.DataFrame(reference)
    pair = reference.loc[rows]
    attributes = [a for a in reference.columns if pair[a].notna().all()]
    if len(attributes) < 3:
        raise ValueError(
            "Cần ít nhất 3 thuộc tính có đủ dữ liệu cho cả hai phim "
            f"(hiện có: {', '.join(attributes) or 'không có'})."
        )

    # 3. Min-Max Scaling về [0, 1] dựa trên toàn bộ dữ liệu tham chiếu
    scaler = MinMaxScaler()
    scaler.fit(reference[attributes].dropna(how="all"))
    movies_scaled = scaler.transform(pair[attributes])
    constant = scaler.data_range_ == 0
    movies_scaled[:, constant] = 0.5

    # 4. Góc các đỉnh radar + đóng vòng tròn
    angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
    angles += angles[:1]

    # 5. Vẽ. Radar thể hiện vị trí tương đối; giá trị gốc kèm đơn vị được
    # Streamlit hiển thị cạnh biểu đồ để người dùng không nhầm hai thang đo.
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    labels = []
    for index, name in zip(rows, (movie1_name, movie2_name)):
        year = df.at[index, "Year"] if "Year" in df.columns else np.nan
        if pd.notna(year):
            try:
                labels.append(f"{name} ({int(float(year))})")
            except (TypeError, ValueError):
                labels.append(str(name))
        else:
            labels.append(str(name))

    for row, label in zip(movies_scaled, labels):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.25)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8)
    ax.set_rlabel_position(90)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_RADAR_LABELS[a] for a in attributes])
    ax.set_title("So sánh tương đối giữa hai phim", pad=30, fontsize=14, fontweight="semibold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.text(
        0.5,
        0.02,
        "Thang 0–100: chuẩn hóa Min-Max trên toàn bộ dữ liệu phim.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    return fig
