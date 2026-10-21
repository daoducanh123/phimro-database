# """Lớp tương thích ngược: giữ nguyên cách import cũ cho code của Tổ 1.

# Bản gốc của load_and_clean_data nay nằm ở src/data_pipeline.py (hợp đồng
# README mục 6). File này chỉ import lại để `python src/data_clean.py` và
# `from src.data_clean import load_and_clean_data` vẫn hoạt động như trước.
# """

# from src.data_pipeline import load_and_clean_data

# __all__ = ["load_and_clean_data"]


# if __name__ == "__main__":
#     # Chạy trực tiếp: làm sạch file thô mặc định rồi in thống kê nhanh.
#     import pathlib

#     default_raw = pathlib.Path("data/collected/resultsRaw_completed.csv")
#     if not default_raw.exists():
#         alt = pathlib.Path("data/resultsRaw.csv")
#         if alt.exists():
#             default_raw = alt
#     if not default_raw.exists():
#         raise SystemExit(
#             "Không tìm thấy file thô. Hãy chạy collect_data.py trước "
#             "(hoặc đặt data/resultsRaw.csv)."
#         )
#     data = load_and_clean_data(default_raw)
#     print(f"Đã làm sạch {len(data):,} bộ phim, {len(data.columns)} cột.")
#     print("Các cột:", list(data.columns))
