"""Phần II và III: Tạo biểu đồ Matplotlib để hiển thị trên Streamlit."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def plot_histogram(df, column):
    """Tạo biểu đồ phân phối của một cột số.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim đã làm sạch.
        column (str): Tên cột số cần trực quan hóa.

    Trả về khi hoàn thiện:
        matplotlib.figure.Figure: Hình để tầng giao diện dùng st.pyplot.

    Công việc cần triển khai:
        - Kiểm tra cột tồn tại, có kiểu số và còn giá trị hợp lệ.
        - Bỏ giá trị thiếu hoặc vô hạn, chọn số khoảng chia phù hợp.
        - Tạo Figure/Axes riêng bằng matplotlib.pyplot.subplots.
        - Vẽ histogram với tiêu đề, tên trục và đơn vị rõ ràng.
        - Trả về Figure, không gọi plt.show hoặc hàm Streamlit tại đây;
          tầng gọi chịu trách nhiệm đóng hình sau khi sử dụng.
    """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column):
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Cột '{column}' không tồn tại hoặc không phải dữ liệu số.")
        
    data_to_plot = df[column].dropna()
    data_to_plot = data_to_plot[~data_to_plot.isin([np.inf, -np.inf])]
    
    if data_to_plot.empty:
        raise ValueError(f"Không có dữ liệu hợp lệ để vẽ biểu đồ cho cột '{column}'.")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data_to_plot, bins=20, kde=True, color="skyblue", edgecolor="black", ax=ax)
    
    ax.set_title(f"Biểu đồ phân bố (Histogram) của {column}", fontsize=14, fontweight='bold')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Số lượng phim", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

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

    # 5. Vẽ
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for row, label in zip(movies_scaled, (movie1_name, movie2_name)):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.25)

    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    ax.set_title("So sánh hai phim (giá trị đã chuẩn hóa Min-Max về [0, 1])", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig
