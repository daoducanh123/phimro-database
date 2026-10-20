"""Phần III và IV: Phân cụm, giảm chiều và dự đoán doanh thu phim."""

import numbers

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Đặc trưng dùng chung cho K-means và PCA (README mục 5 - Tổ 3).
# Khóa là tên chuẩn theo README mục 3.1; giá trị là các tên cột chấp nhận được
# (tên chuẩn đứng trước, tên rút gọn của dữ liệu thô đứng sau).
_CLUSTER_FEATURES = {
    "Budget_USD": ("Budget_USD", "Budget"),
    "Revenue_USD": ("Revenue_USD", "Revenue"),
    "IMDb_Rating": ("IMDb_Rating", "Rating"),
}


def _prepare_cluster_features(df):
    """Trích đặc trưng số hợp lệ cho K-means/PCA (không sửa df đầu vào).

    Trả về DataFrame mới gồm Budget_USD, Revenue_USD, IMDb_Rating đã ép kiểu
    số, loại hàng thiếu/vô hạn, giữ nguyên chỉ mục gốc.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df phải là pandas.DataFrame.")
    if df.empty:
        raise ValueError("Dữ liệu đầu vào rỗng.")

    columns, missing = [], []
    for canonical, candidates in _CLUSTER_FEATURES.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            missing.append(canonical)
        else:
            columns.append(found)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}.")

    X = (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return X.copy()


def run_kmeans_clustering(df, n_clusters):
    """Phân cụm phim bằng K-means trên các đặc trưng số đã chuẩn hóa.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim đã làm sạch.
        n_clusters (int): Số cụm, từ 1 đến số mẫu hợp lệ.

    Trả về khi hoàn thiện:
        pandas.DataFrame: Bản sao dữ liệu có thêm cột cluster.

    Công việc cần triển khai:
        - Thống nhất danh sách đặc trưng; loại mã phim và nhãn cụm cũ.
        - Kiểm tra số cụm, dữ liệu rỗng và số mẫu phân biệt khả dụng.
        - Xử lý giá trị thiếu, vô hạn và chuẩn hóa bằng StandardScaler.
        - Dùng sklearn.cluster.KMeans; đặt random_state và n_init rõ ràng
          để kết quả có thể tái lập.
        - Gắn nhãn đúng chỉ mục ban đầu; quy định cách biểu diễn các hàng
          không thể phân cụm và không thay đổi dữ liệu đầu vào.

    Ghi chú cài đặt (Tổ 3):
        - Đặc trưng: Budget_USD, Revenue_USD, IMDb_Rating (chấp nhận cột
          rút gọn Budget, Revenue, Rating nếu dữ liệu thô chưa đổi tên).
        - Trả về bảng gồm 3 đặc trưng + cột "Cluster" (nhãn 0, 1, 2...),
          giữ chỉ mục gốc. Các hàng thiếu/vô hạn không thể phân cụm bị loại
          khỏi kết quả (không xuất hiện trong bảng trả về).
        - Chuẩn hóa bằng StandardScaler; KMeans(random_state=42, n_init=10).
        - Ném ValueError nếu n_clusters không hợp lệ hoặc lớn hơn số mẫu
          phân biệt khả dụng.
    """
    X = _prepare_cluster_features(df)

    if isinstance(n_clusters, bool) or not isinstance(n_clusters, numbers.Integral):
        raise ValueError("n_clusters phải là số nguyên.")
    n_clusters = int(n_clusters)
    if n_clusters < 1:
        raise ValueError("n_clusters phải >= 1.")
    if len(X) == 0:
        raise ValueError("Không còn dòng hợp lệ sau khi loại giá trị thiếu.")
    if n_clusters > len(X):
        raise ValueError(
            f"n_clusters={n_clusters} lớn hơn số mẫu hợp lệ ({len(X)})."
        )
    n_distinct = len(X.drop_duplicates())
    if n_clusters > n_distinct:
        raise ValueError(
            f"n_clusters={n_clusters} lớn hơn số mẫu phân biệt ({n_distinct})."
        )

    # Đưa tiền và điểm số về cùng thang đo trước khi phân cụm
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    X["Cluster"] = kmeans.fit_predict(X_scaled)

    return X


def run_pca_reduction(df):
    """Giảm đặc trưng phim xuống hai chiều để trực quan hóa.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim, có thể chứa nhãn cluster.

    Trả về khi hoàn thiện:
        pandas.DataFrame: Tọa độ PC1 và PC2, giữ chỉ mục của mẫu hợp lệ.

    Công việc cần triển khai:
        - Chọn đặc trưng số tương ứng với bước phân cụm; loại mã phim
          và cột cluster để nhãn cụm không trở thành đặc trưng.
        - Xử lý giá trị thiếu, vô hạn và chuẩn hóa các đặc trưng.
        - Kiểm tra có ít nhất hai mẫu và hai đặc trưng khả dụng.
        - Dùng sklearn.decomposition.PCA với n_components=2.
        - Giữ liên kết chỉ mục để ghép tọa độ với tên phim và nhãn cụm;
          xem xét tỷ lệ phương sai giải thích khi diễn giải biểu đồ.

    Ghi chú cài đặt (Tổ 3):
        - Nhận df, tự chuẩn hóa bên trong bằng StandardScaler rồi chạy
          PCA(n_components=2, random_state=42) trên Budget_USD, Revenue_USD,
          IMDb_Rating (cột Cluster, nếu có, không được dùng làm đặc trưng).
        - Hai cột kết quả: "Trục X (PCA 1)" và "Trục Y (PCA 2)", giữ chỉ mục
          gốc của các dòng hợp lệ để ghép với Title/Cluster.
        - Tỷ lệ phương sai giải thích lưu ở
          df_pca.attrs["explained_variance_ratio"] (mảng 2 phần tử).
    """
    X = _prepare_cluster_features(df)
    if len(X) < 2:
        raise ValueError("Cần ít nhất 2 dòng hợp lệ để chạy PCA.")
    if X.shape[1] < 2:
        raise ValueError("Cần ít nhất 2 đặc trưng để chạy PCA.")

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(
        data=principal_components,
        columns=["Trục X (PCA 1)", "Trục Y (PCA 2)"],
        index=X.index,  # GIỮ chỉ mục để ghép tên phim / nhãn cụm
    )
    df_pca.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    return df_pca


def train_revenue_prediction_model(df):
    """Huấn luyện mô hình hồi quy để dự đoán doanh thu phòng vé.

    Tham số:
        df (pandas.DataFrame): Dữ liệu có doanh thu mục tiêu và đặc trưng.

    Trả về khi hoàn thiện:
        sklearn.pipeline.Pipeline: Bộ tiền xử lý và mô hình đã huấn luyện,
        hỗ trợ predict trên bảng đặc trưng có cùng cấu trúc đầu vào.

    Công việc cần triển khai:
        - Ánh xạ cột doanh thu thực tế sang biến mục tiêu; thống nhất đơn vị.
        - Loại hàng thiếu mục tiêu; chọn đặc trưng có sẵn tại lúc dự đoán,
          không dùng doanh thu hoặc thông tin phát sinh từ doanh thu.
        - Chia tập huấn luyện/kiểm tra trước khi học bước tiền xử lý;
          đặt random_state để tái lập và kiểm tra số mẫu tối thiểu.
        - Dùng Pipeline và ColumnTransformer để điền khuyết, mã hóa cột
          phân loại và chuẩn hóa khi cần; chỉ fit trên tập huấn luyện.
        - Chọn LinearRegression hoặc RandomForestRegressor.
        - Đánh giá MAE, RMSE và R² trên tập kiểm tra; ghi nhận kết quả
          phục vụ báo cáo, rồi trả về Pipeline đã huấn luyện.

    Ghi chú cài đặt (Phần IV):
        - Mục tiêu: Revenue_USD (chấp nhận cột rút gọn Revenue), đơn vị USD.
        - Đặc trưng: Budget_USD, IMDb_Rating, Votes, Duration_min (số) và
          Genre (phân loại); cột nào không có trong df sẽ được bỏ qua, cần
          ít nhất một đặc trưng. Không dùng Title hay thông tin từ doanh thu.
        - Chia train/test (80/20, random_state=42) rồi mới fit Pipeline.
        - Số: điền khuyết bằng median + StandardScaler. Thể loại: điền
          khuyết bằng giá trị phổ biến nhất + OneHotEncoder (gộp thể loại
          hiếm, bỏ qua thể loại lạ khi dự đoán).
        - Mô hình: RandomForestRegressor(n_estimators=200, random_state=42).
        - Kết quả đánh giá lưu ở pipeline.metrics_ (dict MAE, RMSE, R2,
          n_train, n_test) và danh sách cột đầu vào ở pipeline.feature_columns_.
    """
    numeric_map = {
        "Budget_USD": ("Budget_USD", "Budget"),
        "IMDb_Rating": ("IMDb_Rating", "Rating"),
        "Votes": ("Votes",),
        "Duration_min": ("Duration_min", "Duration"),
    }
    categorical_map = {"Genre": ("Genre",)}

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df phải là pandas.DataFrame.")
    if df.empty:
        raise ValueError("Dữ liệu đầu vào rỗng.")

    def _find(candidates):
        return next((c for c in candidates if c in df.columns), None)

    target_col = _find(("Revenue_USD", "Revenue"))
    if target_col is None:
        raise ValueError("Thiếu cột mục tiêu Revenue_USD.")

    # Đặt tên đầu vào thống nhất (tên chuẩn) cho Pipeline
    data = pd.DataFrame(index=df.index)
    numeric_cols, categorical_cols = [], []
    for name, candidates in numeric_map.items():
        col = _find(candidates)
        if col is not None:
            data[name] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            numeric_cols.append(name)
    for name, candidates in categorical_map.items():
        col = _find(candidates)
        if col is not None:
            data[name] = df[col].astype("object").where(df[col].notna(), np.nan)
            categorical_cols.append(name)
    if not numeric_cols and not categorical_cols:
        raise ValueError("Không có đặc trưng nào khả dụng để huấn luyện.")

    # --- Feature engineering: log-transform + tương tác ---
    if "Budget_USD" in data.columns:
        data["log_Budget_USD"] = np.log1p(data["Budget_USD"].fillna(0))
        numeric_cols.append("log_Budget_USD")
    if "Votes" in data.columns:
        data["log_Votes"] = np.log1p(data["Votes"].fillna(0))
        numeric_cols.append("log_Votes")
    if "Budget_USD" in data.columns and "IMDb_Rating" in data.columns:
        data["Budget_x_Rating"] = data["Budget_USD"] * data["IMDb_Rating"]
        numeric_cols.append("Budget_x_Rating")

    y = (
        pd.to_numeric(df[target_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    valid = y.notna()  # không điền doanh thu thiếu bằng giá trị suy đoán
    X, y = data.loc[valid], y.loc[valid]
    if len(X) < 10:
        raise ValueError(
            f"Cần ít nhất 10 dòng có doanh thu hợp lệ (hiện có {len(X)})."
        )

    # Chia train/test TRƯỚC khi học bước tiền xử lý (tránh data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=5,
                )),
            ]),
            categorical_cols,
        ))
    preprocessor = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1,
            max_depth=15, min_samples_leaf=3, max_features="sqrt"
        )),
    ])
    model.fit(X_train, y_train)

    # Đánh giá trên tập kiểm tra (thang USD gốc)
    y_pred = model.predict(X_test)
    model.metrics_ = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Baseline đơn giản: dự đoán trung bình / trung vị doanh thu
    baseline_mean = np.full_like(y_test, np.mean(y_train))
    baseline_median = np.full_like(y_test, np.median(y_train))
    model.baseline_ = {
        "mean_R2": float(r2_score(y_test, baseline_mean)),
        "median_R2": float(r2_score(y_test, baseline_median)),
        "mean_MAE": float(mean_absolute_error(y_test, baseline_mean)),
        "median_MAE": float(mean_absolute_error(y_test, baseline_median)),
    }

    model.feature_columns_ = numeric_cols + categorical_cols
    model.target_log_transformed_ = False
    return model


def predict_revenue(model, X_raw):
    """Dự đoán doanh thu từ DataFrame đặc trưng thô (chưa biến đổi).

    Áp dụng cùng các biến đổi log và tương tác như lúc huấn luyện,
    rồi gọi model.predict() để trả về doanh thu (USD).

    Tham số:
        model: Pipeline đã huấn luyện từ train_revenue_prediction_model.
        X_raw: pandas.DataFrame với cột thô (Budget_USD, IMDb_Rating,
               Duration_min, Genre, Votes...).

    Trả về:
        numpy.ndarray: Doanh thu dự đoán (USD).
    """
    if not isinstance(X_raw, pd.DataFrame):
        raise TypeError("X_raw phải là pandas.DataFrame.")

    X = X_raw.copy()
    if "Budget_USD" in X.columns:
        X["log_Budget_USD"] = np.log1p(X["Budget_USD"].fillna(0))
    if "Votes" in X.columns:
        X["log_Votes"] = np.log1p(X["Votes"].fillna(0))
    if "Budget_USD" in X.columns and "IMDb_Rating" in X.columns:
        X["Budget_x_Rating"] = X["Budget_USD"] * X["IMDb_Rating"]

    # Chỉ giữ các cột mà mô hình mong đợi
    missing = [c for c in model.feature_columns_ if c not in X.columns]
    if missing:
        raise ValueError(f"Thiếu cột đặc trưng: {', '.join(missing)}")
    X = X[model.feature_columns_]

    return model.predict(X)