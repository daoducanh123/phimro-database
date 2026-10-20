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
    predict_revenue,
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


def format_usd(value):
    """Định dạng tiền USD ngắn gọn, phù hợp phần dữ liệu gốc của radar."""
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return "Không có dữ liệu"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}".replace(".", ",") + " tỷ USD"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.0f}".replace(",", ".") + " triệu USD"
    return f"{value:,.0f}".replace(",", ".") + " USD"


def format_year(value):
    """Định dạng năm phát hành nếu dữ liệu có sẵn."""
    value = pd.to_numeric(value, errors="coerce")
    return str(int(value)) if pd.notna(value) else "Không rõ năm"



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
        "Phần I: Tổng quan & Xử lý Dữ liệu",
        "Phần II: Thống kê & Phân bố",
        "Phần III: Học máy - Phân cụm & So sánh",
        "Phần IV: Dự đoán Doanh thu phòng vé",
    ),
)

if section == "Phần I: Tổng quan & Xử lý Dữ liệu":
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

elif section == "Phần II: Thống kê & Phân bố":
    st.header("Phần II: Thống kê và phân phối")
    
    # 1. Lấy dữ liệu từ session (nếu đã làm sạch ở Phần I)
    df = get_data()
    
    if df is None:
        st.error("Chưa có dữ liệu. Hãy quay lại 'Phần I' để chạy làm sạch hoặc tải file lên trước.")
    else:
            # Cột số khả dụng cho thống kê
            stat_cols = [c for c in ("Duration_min", "IMDb_Rating", "Budget_USD", "Revenue_USD") if c in df.columns]
            if not stat_cols:
                st.warning("Không tìm thấy cột số nào để thống kê trong dữ liệu.")
            else:
                # --- 1. Bảng thống kê cơ bản ---
                st.subheader("1. Thống kê mô tả cơ bản")
                try:
                    st.subheader("Thống kê chung")
                    stats_overall, stats_by_genre = calculate_basic_stats(df)
                    st.dataframe(stats_overall, use_container_width=True)
                    if not stats_by_genre.empty:
                        st.subheader("Thống kê theo thể loại")
                        st.dataframe(stats_by_genre, use_container_width=True)
                    # Lưu hai artifact: thống kê chung và bảng genre phẳng,
                    # dùng trực tiếp được trong Excel/báo cáo.
                    stats_by_genre.to_csv(
                        "data/results2_by_genre.csv",
                        index=True,
                        encoding="utf-8-sig",
                    )
                    try:
                        stats_overall.to_csv(
                            "data/results2.csv", index=True, encoding="utf-8-sig"
                        )
                        st.caption(
                            "Đã lưu data/results2.csv và "
                            "data/results2_by_genre.csv"
                        )
                    except Exception as exc:
                        st.warning(f"Không lưu được data/results2.csv: {exc}")
                except Exception as exc:
                    st.error(f"Lỗi khi tính thống kê cơ bản: {exc}")

                st.divider()

                # --- 2. Phim cao nhất / thấp nhất theo chỉ số ---
                st.subheader("2. Phim cao nhất / thấp nhất theo chỉ số")
                column = st.selectbox("Chọn chỉ số", stat_cols)
                try:
                    top_movies, bottom_movies = get_top_bottom_movies(
                        df, column, top_n=3
                    )
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader(f"Top 3 cao nhất ({column})")
                        st.dataframe(top_movies, use_container_width=True)
                    with col_b:
                        st.subheader(f"Top 3 thấp nhất ({column})")
                        st.dataframe(bottom_movies, use_container_width=True)
                except Exception as exc:
                    st.error(f"Lỗi khi tìm phim cực trị: {exc}")

                st.divider()

                # --- 3. Biểu đồ phân phối ---
                st.subheader(f"3. Biểu đồ phân phối: {column}")
                try:
                    fig = plot_histogram(df, column)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Lỗi khi vẽ biểu đồ: {exc}")


elif section == "Phần III: Học máy - Phân cụm & So sánh":
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
            st.subheader("1. Kết quả phân cụm")

            def find_cluster_column(*candidates):
                return next((col for col in candidates if col in clustered.columns), None)

            rating_col = find_cluster_column("Rating", "IMDb_Rating")
            budget_col = find_cluster_column("Budget", "Budget_USD")
            revenue_col = find_cluster_column("Revenue", "Revenue_USD")
            cluster_summary = clustered.groupby("Cluster").agg(
                Số_phim=("Cluster", "size"),
                Rating_trung_vị=(rating_col, "median"),
                Kinh_phí_trung_vị=(budget_col, "median"),
                Doanh_thu_trung_vị=(revenue_col, "median"),
            ).reset_index()
            cluster_summary["Rating_trung_vị"] = cluster_summary[
                "Rating_trung_vị"
            ].map(lambda value: f"{value:.1f}".replace(".", ",") + " / 10")
            for column in ("Kinh_phí_trung_vị", "Doanh_thu_trung_vị"):
                cluster_summary[column] = cluster_summary[column].map(format_usd)

            # Bản đồ cụm 2D sau khi giảm chiều PCA
            ev = pca_coords.attrs["explained_variance_ratio"]
            fig, ax = plt.subplots(figsize=(8, 5.5))
            palette = plt.get_cmap("tab10", n_clusters)
            for cluster_id in sorted(clustered["Cluster"].unique()):
                mask = clustered.loc[pca_coords.index, "Cluster"] == cluster_id
                ax.scatter(
                    pca_coords.loc[mask, pca_coords.columns[0]],
                    pca_coords.loc[mask, pca_coords.columns[1]],
                    color=palette(cluster_id),
                    s=16,
                    alpha=0.72,
                    label=f"Cụm {cluster_id} · {int(mask.sum()):,} phim",
                )
            ax.set_xlabel(f"PCA 1 · giải thích {ev[0]:.1%} phương sai")
            ax.set_ylabel(f"PCA 2 · giải thích {ev[1]:.1%} phương sai")
            ax.set_title(f"Bản đồ phân cụm phim (K = {n_clusters})")
            ax.grid(alpha=0.25)
            ax.legend(title="Nhóm phim", frameon=False, loc="best")
            fig.tight_layout()

            map_column, summary_column = st.columns([1.2, 0.8])
            with map_column:
                st.caption(
                    "Mỗi điểm là một phim; vị trí gần nhau biểu thị hồ sơ "
                    "Budget, Revenue và Rating tương đồng."
                )
                st.pyplot(fig, use_container_width=True)
            with summary_column:
                st.caption("Tóm tắt theo cụm bằng số liệu gốc (trung vị).")
                st.dataframe(cluster_summary, hide_index=True, use_container_width=True)
            plt.close(fig)

            with st.expander("Xem bảng dữ liệu chi tiết (50 dòng đầu)"):
                st.dataframe(table.head(50), use_container_width=True)


            st.subheader("2. So sánh hai phim bằng biểu đồ radar")

            radar_df = df.dropna(
                subset=["IMDb_Rating", "Budget_USD", "Revenue_USD", "Duration_min"]
            ).copy()

            # Hàm radar chủ động báo lỗi với title trùng; loại khỏi danh sách chọn UI.
            duplicate_titles = radar_df["Title"].duplicated(keep=False)
            movie_options = sorted(
                radar_df.loc[~duplicate_titles, "Title"]
                .astype(str)
                .str.strip()
                .unique()
            )

            col1, col2 = st.columns(2)
            with col1:
                movie1 = st.selectbox("Phim thứ nhất", movie_options, key="radar_movie_1")
            with col2:
                movie2 = st.selectbox(
                    "Phim thứ hai",
                    movie_options,
                    index=1,
                    key="radar_movie_2",
                )

            if movie1 == movie2:
                st.warning("Hãy chọn hai phim khác nhau.")
            else:
                try:
                    fig = plot_radar_chart(df, movie1, movie2)
                    chart_column, detail_column = st.columns([1.2, 0.8])
                    with chart_column:
                        st.caption(
                            "Radar thể hiện mức tương đối 0–100. "
                            "Xem số liệu gốc có đơn vị ở bên phải."
                        )
                        st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    with detail_column:
                        # st.metric có cỡ chữ mặc định lớn cho cột hẹp của
                        # phần radar; thu nhỏ và cho phép giá trị dài xuống dòng.
                        st.markdown(
                            """
                            <style>
                            [data-testid="stMetricValue"] {
                                font-size: clamp(1.25rem, 2.8vw, 1.65rem);
                                white-space: normal;
                                overflow-wrap: anywhere;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("#### Dữ liệu gốc")
                        selected_rows = []
                        titles = radar_df["Title"].astype(str).str.strip()
                        for name in (movie1, movie2):
                            selected_rows.append(
                                radar_df.loc[titles == name].iloc[0]
                            )

                        for icon, movie, row in zip(
                            ("🔵", "🔴"),
                            (movie1, movie2),
                            selected_rows,
                        ):
                            st.markdown(
                                f"**{icon} {movie} ({format_year(row.get('Year'))})**"
                            )
                            metric_left, metric_right = st.columns(2)
                            metric_left.metric(
                                "Điểm TMDB",
                                f"{row['IMDb_Rating']:.1f}".replace(".", ",") + " / 10",
                            )
                            metric_right.metric(
                                "Thời lượng",
                                f"{row['Duration_min']:,.0f}".replace(",", ".") + " phút",
                            )
                            metric_left, metric_right = st.columns(2)
                            metric_left.metric("Kinh phí", format_usd(row["Budget_USD"]))
                            metric_right.metric("Doanh thu", format_usd(row["Revenue_USD"]))
                            st.divider()
                except ValueError as exc:
                    st.error(str(exc))


elif section == "Phần IV: Dự đoán Doanh thu phòng vé":
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
            if hasattr(model, "baseline_") and model.baseline_:
                b = model.baseline_
                st.caption(
                    f"Baseline: R² trung bình = {b['mean_R2']:.3f}, "
                    f"R² trung vị = {b['median_R2']:.3f} "
                    f"(mô hình R² = {metrics['R2']:.3f})"
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
                if "Votes" in model.feature_columns_ or "log_Votes" in model.feature_columns_:
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
                # Dự đoán qua predict_revenue (xử lý log-transform nội bộ)
                prediction = float(predict_revenue(model, pd.DataFrame([row]))[0])
                st.success(
                    f"Doanh thu dự đoán: **${prediction:,.0f}** USD"
                )
