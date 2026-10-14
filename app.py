"""Giao diện chính cho ứng dụng phân tích và dự đoán dữ liệu phim."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_pipeline import load_and_clean_data
from src.models import (
    run_kmeans_clustering,
    run_pca_reduction,
    train_revenue_prediction_model,
)
from src.statistics_eda import calculate_basic_stats, get_top_bottom_movies
from src.visualizer import plot_histogram, plot_radar_chart

# Đường dẫn dữ liệu sạch (Tổ 1 xuất ra), tính từ vị trí file app.py
BASE_DIR = Path(__file__).resolve().parent
CLEANED_CSV = BASE_DIR / "data" / "cleaned" / "results.csv"


@st.cache_data
def load_cleaned_data():
    """Đọc dữ liệu sạch cho các trang; trả về None nếu chưa có tệp.

    Kết quả được cache để không đọc lại CSV sau mỗi lần tương tác.
    """
    if not CLEANED_CSV.exists():
        return None
    return pd.read_csv(CLEANED_CSV)


def get_data():
    """Lấy dữ liệu để phân tích: ưu tiên bảng vừa làm sạch trong session.

    Trả về None nếu người dùng chưa tải/làm sạch và cũng chưa có
    data/cleaned/results.csv.
    """
    if "df" in st.session_state:
        return st.session_state["df"]
    return load_cleaned_data()



# Các hàm khác được nhập sẵn để từng nhóm tích hợp khi hoàn thành.
st.set_page_config(
    page_title="Movie Data Analysis & Prediction",
    page_icon="🎬",
    layout="wide",
)

st.title("Phân tích dữ liệu và dự đoán doanh thu phim")
st.caption("Dự án phân tích dữ liệu IMDb/TMDB")

section = st.sidebar.radio(
    "Điều hướng",
    (
        "Overview & Data Cleaning (Part I)",
        "Statistics & Distribution (Part II)",
        "Clustering & Radar Chart (Part III)",
        "Revenue Prediction (Part IV)",
    ),
)

if section == "Overview & Data Cleaning (Part I)":
    st.header("Phần I: Tổng quan và làm sạch dữ liệu")

    # Nút làm sạch file thô mặc định có sẵn trong repo (không cần upload)
    default_raw = BASE_DIR / "data" / "collected" / "resultsRaw_completed.csv"
    if not default_raw.exists():
        default_raw = BASE_DIR / "data" / "resultsRaw.csv"

    st.subheader("1. Làm sạch từ file thô trong dự án")
    if default_raw.exists():
        st.caption(
            f"File thô: data/{default_raw.parent.name}/{default_raw.name}"
        )
        if st.button("Chạy làm sạch dữ liệu"):
            with st.spinner("Đang làm sạch..."):
                try:
                    st.session_state["df"] = load_and_clean_data(default_raw)
                    st.success(
                        "Đã làm sạch và lưu ra data/cleaned/results.csv!"
                    )
                except Exception as exc:
                    st.error(f"Làm sạch thất bại: {exc}")
    else:
        st.warning("Chưa có file thô. Hãy chạy `python src/collect_data.py`.")

    # Hoặc tự tải file thô của riêng mình lên
    st.subheader("2. Hoặc tải file thô của bạn (CSV)")
    raw_file = st.file_uploader("Chọn file dữ liệu thô (CSV)", type=["csv"])
    if raw_file is not None:
        try:
            st.session_state["df"] = load_and_clean_data(raw_file)
            st.success("Đã làm sạch file tải lên!")
        except Exception as exc:
            st.error(f"Không đọc được file: {exc}")

    # Hiển thị kết quả nếu đã có dữ liệu sạch
    df_raw = st.session_state.get("df")
    if df_raw is None:
        df_raw = load_cleaned_data()
    if df_raw is None:
        st.info("Chưa có dữ liệu. Hãy chạy làm sạch hoặc tải file thô lên.")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Số bộ phim", f"{len(df_raw):,}")
        col_b.metric("Số cột", len(df_raw.columns))
        col_c.metric("Cột số", df_raw.select_dtypes("number").shape[1])
        st.subheader("Xem trước dữ liệu đã làm sạch")
        st.dataframe(df_raw.head(50), use_container_width=True)

elif section == "Statistics & Distribution (Part II)":
    st.header("Phần II: Thống kê và phân phối")
    
    # 1. Lấy dữ liệu từ session (nếu đã làm sạch ở Phần I)
    df = get_data()
    
    if df is None:
        st.error("Chưa có dữ liệu. Hãy quay lại 'Phần I' để chạy làm sạch hoặc tải file lên trước.")
    else:
        st.info(
            "Nhóm Phần II: Tích hợp calculate_basic_stats(df), "
            "get_top_bottom_movies(df, column) và plot_histogram(df, column). "
            "Lưu bảng thống kê vào data/results2.csv và dùng st.pyplot cho hình."
        )
        
        # ==========================================
        # PHẦN CODE CỦA BẠN (VẼ HISTOGRAM)
        # ==========================================
        st.subheader("Biểu đồ Phân phối (Histogram)")
        
        # Tạo dropdown chọn Điểm số hoặc Doanh thu
        option = st.selectbox(
            "Chọn dữ liệu muốn xem:", 
            options=["Điểm số (IMDb_Rating)", "Doanh thu (Revenue_USD)"]
        )
        
        # Đưa về đúng tên cột chuẩn trong file cleaned data của dự án
        col_to_plot = "Rating" if option == "Điểm số (IMDb_Rating)" else "Revenue"

        # Thêm 2 dòng này ngay bên dưới để đảm bảo dữ liệu là số, đề phòng file CSV bị lỗi định dạng chữ:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
        
        try:
            # Gọi hàm vẽ bạn đã viết ở src/visualizer.py
            fig_hist = plot_histogram(df, col_to_plot)
            st.pyplot(fig_hist)
        except Exception as e:
            st.error(f"Lỗi khi vẽ biểu đồ: {e}")


elif section == "Clustering & Radar Chart (Part III)":
    st.header("Phần III: Phân cụm và biểu đồ radar")
    df = load_cleaned_data()
    if df is None:
        st.error(
            "Chưa tìm thấy data/cleaned/results.csv. "
            "Hãy chạy phần làm sạch dữ liệu trước."
        )
    else:
        n_clusters = st.slider(
            "Số cụm (n_clusters)", 2, 10, 3, key="n_clusters"
        )
        try:
            clustered = run_kmeans_clustering(df, n_clusters)
            pca_coords = run_pca_reduction(df)
        except ValueError as exc:
            st.error(f"Không chạy được phân cụm: {exc}")
        else:
            # Gộp tên phim + nhãn cụm + tọa độ PCA (khóa chung: index)
            table = (
                df.loc[clustered.index, ["Title"]]
                .join(clustered)
                .join(pca_coords)
            )
            counts = (
                clustered["Cluster"]
                .value_counts()
                .sort_index()
                .rename_axis("Cluster")
                .reset_index(name="Số phim")
            )
            st.subheader("Kết quả phân cụm")
            st.dataframe(counts, use_container_width=True)
            st.bar_chart(counts.set_index("Cluster"))

            # Bản đồ cụm 2D sau khi giảm chiều PCA
            ev = pca_coords.attrs["explained_variance_ratio"]
            fig, ax = plt.subplots(figsize=(8, 5))
            points = ax.scatter(
                pca_coords.iloc[:, 0],
                pca_coords.iloc[:, 1],
                c=clustered.loc[pca_coords.index, "Cluster"],
                cmap="tab10",
                s=14,
                alpha=0.75,
            )
            ax.set_xlabel(f"Trục X (PCA 1) — {ev[0]:.1%} phương sai")
            ax.set_ylabel(f"Trục Y (PCA 2) — {ev[1]:.1%} phương sai")
            ax.set_title(f"Bản đồ cụm K-means (K={n_clusters})")
            fig.colorbar(points, ax=ax, label="Cluster")
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Xem bảng dữ liệu chi tiết (50 dòng đầu)"):
                st.dataframe(table.head(50), use_container_width=True)

            st.info(
                "plot_radar_chart(df, movie1_name, movie2_name) đã "
                "được Tổ 3 cài xong — Nhóm trưởng sẽ nối vào đây "
                "sau khi rà soát."
            )

elif section == "Revenue Prediction (Part IV)":
    st.header("Phần IV: Dự đoán doanh thu")
    df = load_cleaned_data()
    if df is None:
        st.error(
            "Chưa tìm thấy data/cleaned/results.csv. "
            "Hãy chạy phần làm sạch dữ liệu trước."
        )
    else:
        # Chỉ huấn luyện khi bấm nút; kết quả giữ trong session
        # để không train lại sau mỗi lần tương tác giao diện.
        if st.button("Huấn luyện mô hình (RandomForest)"):
            with st.spinner("Đang huấn luyện mô hình..."):
                try:
                    model = train_revenue_prediction_model(df)
                except ValueError as exc:
                    st.error(f"Không huấn luyện được: {exc}")
                else:
                    st.session_state["revenue_model"] = model
                    st.success("Huấn luyện hoàn tất!")

        model = st.session_state.get("revenue_model")
        if model is None:
            st.info(
                "Bấm nút bên trên để huấn luyện mô hình. Mô hình "
                "chỉ train một lần cho mỗi phiên làm việc."
            )
        else:
            metrics = model.metrics_
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"${metrics['MAE']:,.0f}")
            col2.metric("RMSE", f"${metrics['RMSE']:,.0f}")
            col3.metric("R²", f"{metrics['R2']:.3f}")
            col4.metric("Số mẫu test", f"{metrics['n_test']:,}")
            st.caption(
                "Đặc trưng đầu vào: " + ", ".join(model.feature_columns_)
            )

            st.subheader("Dự đoán doanh thu cho phim mới")
            with st.form("revenue_form"):
                budget = st.number_input(
                    "Kinh phí (USD)",
                    min_value=0.0,
                    value=10_000_000.0,
                    step=500_000.0,
                )
                rating = st.number_input(
                    "Điểm IMDb", 0.0, 10.0, 7.0, 0.1
                )
                duration = st.number_input(
                    "Thời lượng (phút)", 0, 300, 120
                )
                genre = st.text_input("Thể loại (Genre)", "Drama")
                # Ô Votes chỉ xuất hiện nếu mô hình dùng cột này
                votes = 0
                if "Votes" in model.feature_columns_:
                    votes = st.number_input(
                        "Số lượt đánh giá (Votes)", 0, 10_000_000, 1_000
                    )
                submitted = st.form_submit_button("Dự đoán doanh thu")

            if submitted:
                row = {
                    "Budget_USD": budget,
                    "IMDb_Rating": rating,
                    "Duration_min": duration,
                    "Genre": genre,
                    "Votes": votes,
                }
                # Pipeline yêu cầu tên cột chuẩn nên đổi tên tại đây
                features = pd.DataFrame([row])[model.feature_columns_]
                prediction = float(model.predict(features)[0])
                st.success(
                    f"Doanh thu dự đoán: **${prediction:,.0f}** USD"
                )
