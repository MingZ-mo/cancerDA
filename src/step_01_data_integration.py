import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DATA_RAW_DIR = "CI5plus_Detailed"
DATA_CLEAN_DIR = "cleanedData"
OUTPUT_DIR = "cleanedData"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def kiem_ke_tap_tin_tho_va_tien_xu_ly():
    print("=" * 85)
    print("TASK 1.1: KIỂM KÊ CẤU TRÚC CHI TIẾT CÁC TỆP DỮ LIỆU UNG THƯ (DATA AUDIT)")
    print("=" * 85)
    
    # 1. Kiểm kê tệp thô
    print("[1] CÁC TỆP DỮ LIỆU GỐC (IARC CI5plus):")
    if os.path.exists(DATA_RAW_DIR):
        raw_files = sorted(os.listdir(DATA_RAW_DIR))
        for i, fn in enumerate(raw_files, 1):
            path = os.path.join(DATA_RAW_DIR, fn)
            size = os.path.getsize(path)
            print(f"    [{i:02d}] {fn:<25} | Kích thước: {size:>12,} bytes ({size/(1024*1024):.2f} MB)")
            
    # 2. Kiểm kê tệp sạch
    print("\n[2] CÁC TỆP DỮ LIỆU ĐÃ TIỀN XỬ LÝ (cleanedData/):")
    if os.path.exists(DATA_CLEAN_DIR):
        clean_files = sorted(os.listdir(DATA_CLEAN_DIR))
        for i, fn in enumerate(clean_files, 1):
            path = os.path.join(DATA_CLEAN_DIR, fn)
            size = os.path.getsize(path)
            print(f"    [{i:02d}] {fn:<40} | Kích thước: {size:>12,} bytes ({size/(1024*1024):.2f} MB)")
    print("-" * 85)

def tich_hop_va_tao_tap_du_lieu_phan_tich():
    print("=" * 85)
    print("TASK 1.2: TÍCH HỢP ĐA NGUỒN & XÂY DỰNG PANEL DATASET DỊCH TỄ HỌC UNG THƯ")
    print("=" * 85)

    # 1. Nguồn 1: Bảng tổng hợp theo Quốc gia & Châu lục (eda_continent_country_summary.csv)
    path_country = os.path.join(DATA_CLEAN_DIR, "eda_continent_country_summary.csv")
    print(f"- Đang nạp dữ liệu quốc gia từ: {path_country}")
    df_country = pd.read_csv(path_country)

    # 2. Lọc các vị trí ung thư tiêu biểu phục vụ phân tích đa biến
    # Mã 1: All cancers, 46: Lung, 67: Breast, 101: Prostate, 21: Colon, 22: Rectum, 19: Stomach, 31: Liver
    cancer_pivots = {
        1: "All_Cancers_Rate",
        46: "Lung_Rate",
        67: "Breast_Rate",
        101: "Prostate_Rate",
        21: "Colon_Rate",
        22: "Rectum_Rate",
        19: "Stomach_Rate",
        31: "Liver_Rate"
    }

    df_sub = df_country[df_country["cancer_code"].isin(cancer_pivots.keys())].copy()
    df_sub["cancer_name"] = df_sub["cancer_code"].map(cancer_pivots)

    # Pivot table để mỗi dòng là: continent, country, year, sex với các cột tỷ lệ ung thư
    df_pivot_rates = df_sub.pivot_table(
        index=["continent", "country", "year", "sex"],
        columns="cancer_name",
        values="crude_rate_per_100k",
        aggfunc="sum"
    ).reset_index()

    # Tổng hợp Cases và PY cho All Cancers (cancer_code == 1)
    df_all = df_country[df_country["cancer_code"] == 1][
        ["continent", "country", "year", "sex", "cases", "py", "crude_rate_per_100k"]
    ].rename(columns={
        "cases": "Total_Cases",
        "py": "Total_Population",
        "crude_rate_per_100k": "Overall_Crude_Rate"
    })

    # Ghép bảng tổng hợp và bảng pivot rates
    df_panel = pd.merge(df_all, df_pivot_rates, on=["continent", "country", "year", "sex"], how="left")
    
    # Tính gộp Colorectal Rate = Colon + Rectum
    if "Colon_Rate" in df_panel.columns and "Rectum_Rate" in df_panel.columns:
        df_panel["Colorectal_Rate"] = np.round(df_panel["Colon_Rate"].fillna(0) + df_panel["Rectum_Rate"].fillna(0), 2)

    # Điền giá trị khuyết cho các ung thư đặc thù giới tính:
    # Nam không có Breast Rate đáng kể và Nữ không có Prostate Rate (điền 0)
    for col in cancer_pivots.values():
        if col in df_panel.columns:
            df_panel[col] = df_panel[col].fillna(0.0)

    # 3. Tạo biến trễ thời gian (Lag 1) cho biến mục tiêu Overall_Crude_Rate theo từng Quốc gia và Giới tính
    df_panel = df_panel.sort_values(by=["country", "sex", "year"]).reset_index(drop=True)
    df_panel["Rate_Lag1"] = df_panel.groupby(["country", "sex"])["Overall_Crude_Rate"].shift(1)
    
    # Biến đổi Logarit để ổn định phương sai cho phân tích hồi quy sau này
    df_panel["Log_Cases"] = np.round(np.log1p(df_panel["Total_Cases"]), 4)
    df_panel["Log_Population"] = np.round(np.log(np.maximum(df_panel["Total_Population"], 1.0)), 4)
    df_panel["Time_Index"] = df_panel["year"] - 2000

    # 4. Xuất panel dataset chính vào cleanedData/cancer_country_panel.csv
    output_panel_path = os.path.join(OUTPUT_DIR, "cancer_country_panel.csv")
    df_panel.to_csv(output_panel_path, index=False, encoding="utf-8")
    print(f"- Đã xuất bảng Panel đa quốc gia ({len(df_panel):,} dòng, {len(df_panel.columns)} cột) -> {output_panel_path}")

    # 5. Xây dựng thêm tập dữ liệu chuỗi thời gian toàn cầu (Global Time-Series Dataset) từ eda_cancer_yearly_summary.csv
    path_yearly = os.path.join(DATA_CLEAN_DIR, "eda_cancer_yearly_summary.csv")
    df_yearly = pd.read_csv(path_yearly)
    
    # Lọc mã All Cancers (cancer_code == 1)
    df_ts_global = df_yearly[df_yearly["cancer_code"] == 1].sort_values(by=["sex", "year"]).reset_index(drop=True)
    df_ts_global["Time_Index"] = df_ts_global["year"] - 2000
    df_ts_global["ASR_Lag1"] = df_ts_global.groupby("sex")["asr_world_per_100k"].shift(1)
    df_ts_global["Crude_Rate_Lag1"] = df_ts_global.groupby("sex")["crude_rate_per_100k"].shift(1)

    output_ts_path = os.path.join(OUTPUT_DIR, "cancer_global_timeseries.csv")
    df_ts_global.to_csv(output_ts_path, index=False, encoding="utf-8")
    print(f"- Đã xuất chuỗi thời gian toàn cầu ({len(df_ts_global):,} dòng) -> {output_ts_path}")

    # 6. Self-Test Assertions
    print("\n" + "=" * 85)
    print("TASK 1.3: THỰC THI SELF-TEST KIỂM TOÁN TÍNH TOÀN VẸN DỮ LIỆU TÍCH HỢP")
    print("=" * 85)
    assert os.path.exists(output_panel_path), f"Lỗi Self-Test: Không tìm thấy {output_panel_path}"
    assert os.path.exists(output_ts_path), f"Lỗi Self-Test: Không tìm thấy {output_ts_path}"
    assert len(df_panel) > 1000, f"Lỗi Self-Test: Dữ liệu panel quá ít dòng ({len(df_panel)})"
    assert "Overall_Crude_Rate" in df_panel.columns, "Lỗi Self-Test: Thiếu cột Overall_Crude_Rate"
    assert "Rate_Lag1" in df_panel.columns, "Lỗi Self-Test: Thiếu cột Rate_Lag1"
    assert (df_panel["Total_Cases"] >= 0).all(), "Lỗi Self-Test: Phát hiện Total_Cases âm"
    assert (df_panel["Total_Population"] > 0).all(), "Lỗi Self-Test: Phát hiện Total_Population <= 0"
    assert set(df_panel["year"].unique()) == set(range(2000, 2018)), "Lỗi Self-Test: Không đủ các năm 2000-2017"

    print(f"[PASS] Toàn bộ tiêu chí Self-Test Bước 1 đều đạt 100%!")
    print(f"[THỐNG KÊ NHANH] Tổng số quốc gia tham gia tích hợp: {df_panel['country'].nunique()} nước thuộc {df_panel['continent'].nunique()} châu lục.")
    return df_panel, df_ts_global

if __name__ == "__main__":
    kiem_ke_tap_tin_tho_va_tien_xu_ly()
    tich_hop_va_tao_tap_du_lieu_phan_tich()
