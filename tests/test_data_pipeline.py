import pytest
import pandas as pd
import numpy as np
import os

# Import hàm cần test từ mã nguồn của nhóm
from src.data_pipeline import load_and_clean_data

@pytest.fixture
def mock_raw_csv(tmp_path):
    """
    Tạo một file CSV giả lập chứa các trường hợp lỗi thường gặp 
    để dùng chung cho các bài test.
    """
    data = {
        "Title": ["Movie A", "Movie B", "Movie C"],
        "Genre": ["Action", "Comedy", ""], # Dòng C thiếu Genre
        "Budget_USD": [1000000, 0, 500000], # Dòng B có budget = 0
        "Revenue_USD": [2500000, 3000000, 0], # Dòng C có revenue = 0
        "IMDb_Rating": [7.5, 8.0, 0], # Dòng C điểm = 0
        "Duration_min": [120, 90, 150]
    }
    df_mock = pd.DataFrame(data)
    
    # Tạo đường dẫn file tạm thời an toàn
    file_path = tmp_path / "mock_raw_data.csv"
    df_mock.to_csv(file_path, index=False)
    
    return str(file_path)

def test_convert_zero_to_nan(mock_raw_csv):
    """
    Kịch bản 1: Kiểm tra quy tắc giá trị 0 phải bị biến thành NaN 
    đối với các cột kinh tế và điểm số.
    """
    # Chạy hàm làm sạch của dự án
    df_clean = load_and_clean_data(mock_raw_csv)
    
    # Movie B có Budget = 0 ở file gốc, sau khi clean phải thành NaN
    assert pd.isna(df_clean.loc[df_clean["Title"] == "Movie B", "Budget_USD"].values[0]), "Lỗi: Giá trị Budget 0 chưa được chuyển thành NaN"
    
    # Movie C có Revenue = 0 ở file gốc, sau khi clean phải thành NaN
    assert pd.isna(df_clean.loc[df_clean["Title"] == "Movie C", "Revenue_USD"].values[0]), "Lỗi: Giá trị Revenue 0 chưa được chuyển thành NaN"
    
    # Movie C có IMDb_Rating = 0, sau khi clean phải thành NaN
    assert pd.isna(df_clean.loc[df_clean["Title"] == "Movie C", "IMDb_Rating"].values[0]), "Lỗi: Giá trị Rating 0 chưa được chuyển thành NaN"

def test_missing_essential_columns(tmp_path):
    """
    Kịch bản 2: Kiểm tra việc thiếu cột quan trọng (ví dụ: mất cột Budget_USD)
    Hàm làm sạch phải báo lỗi KeyError hoặc ValueError thay vì chạy tiếp ra rác.
    """
    # Tạo data bị thiếu hẳn cột Budget_USD
    bad_data = {
        "Title": ["Movie X"],
        "Genre": ["Horror"],
        "Revenue_USD": [1000]
        # Bị mất cột Budget_USD
    }
    bad_file_path = tmp_path / "bad_data.csv"
    pd.DataFrame(bad_data).to_csv(bad_file_path, index=False)
    
    # Kiểm tra xem hàm có chủ động ném ra lỗi khi thiếu cột không
    with pytest.raises(KeyError):
        load_and_clean_data(str(bad_file_path))