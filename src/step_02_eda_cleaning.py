import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

DATA_PANEL_PATH = os.path.join("cleanedData", "cancer_country_panel.csv")
DATA_AGE_PATH = os.path.join("cleanedData", "eda_age_incidence_pattern.csv")
DATA_YEARLY_PATH = os.path.join("cleanedData", "eda_cancer_yearly_summary.csv")
OUTPUT_CLEANED_PATH = os.path.join("cleanedData", "cancer_country_panel_cleaned.csv")
FIG01_PATH = os.path.join("figures", "fig01_cancer_trends_dual_axis.png")
FIG02_PATH = os.path.join("figures", "fig02_age_incidence_distribution.png")

os.makedirs("figures", exist_ok=True)

def thong_ke_mo_ta_ban_dau(df, cac_cot_dinh_luong):
    print("=" * 85)
    print("TASK 2.1: BẢNG THỐNG KÊ MÔ TẢ TẬP DỮ LIỆU DỊCH TỄ BAN ĐẦU (DESCRIPTIVE STATISTICS)")
    print("=" * 85)
    
    bang_thong_ke = []
    for cot in cac_cot_dinh_luong:
        if cot not in df.columns:
            continue
        chuoi = df[cot].dropna()
        mode_series = chuoi.mode()
        bang_thong_ke.append({
            "Biến số": cot,
            "Số mẫu (N)": int(chuoi.count()),
            "Mean": round(chuoi.mean(), 2),
            "Std": round(chuoi.std(), 2),
            "Median": round(chuoi.median(), 2),
            "Mode": round(mode_series.iloc[0] if len(mode_series) > 0 else np.nan, 2),
            "Min": round(chuoi.min(), 2),
            "Q25 (25%)": round(chuoi.quantile(0.25), 2),
            "Q75 (75%)": round(chuoi.quantile(0.75), 2),
            "Max": round(chuoi.max(), 2)
        })
        
    df_thong_ke = pd.DataFrame(bang_thong_ke)
    print(df_thong_ke.to_string(index=False))
    return df_thong_ke

def xu_ly_trung_lap(df):
    print("\n" + "=" * 85)
    print("TASK 2.2: KIỂM TRA VÀ XỬ LÝ DỮ LIỆU TRÙNG LẶP (DUPLICATE REMOVAL)")
    print("=" * 85)
    
    khoa_chinh = ["country", "year", "sex"]
    so_dong_trung = df.duplicated(subset=khoa_chinh).sum()
    print(f"- Số bản ghi trùng lặp khóa chính {khoa_chinh}: {so_dong_trung} dòng.")
    if so_dong_trung > 0:
        df_khong_trung = df.drop_duplicates(subset=khoa_chinh, keep="first").copy()
    else:
        df_khong_trung = df.copy()
    return df_khong_trung

def xu_ly_gia_tri_khuyet(df, cac_cot_dinh_luong):
    print("\n" + "=" * 85)
    print("TASK 2.3: KIỂM TRA VÀ XỬ LÝ GIÁ TRỊ KHUYẾT (MISSING VALUES IMPUTATION)")
    print("=" * 85)
    
    tong_so_dong = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / tong_so_dong) * 100.0
    
    bang_missing = pd.DataFrame({
        "Cột dữ liệu": df.columns,
        "Số ô trống": missing_count.values,
        "Tỷ lệ thiếu (%)": np.round(missing_pct.values, 2)
    })
    print(bang_missing[bang_missing["Số ô trống"] > 0].to_string(index=False))
    
    df_imputed = df.copy()
    # Rate_Lag1 bị khuyết tại năm đầu tiên (2000) của mỗi quốc gia/giới tính -> Backfill (bfill)
    if "Rate_Lag1" in df_imputed.columns and df_imputed["Rate_Lag1"].isnull().sum() > 0:
        df_imputed["Rate_Lag1"] = df_imputed.groupby(["country", "sex"])["Rate_Lag1"].bfill()
        # Nếu vẫn còn sót do quốc gia chỉ có 1 năm -> điền bằng chính Overall_Crude_Rate
        df_imputed["Rate_Lag1"] = df_imputed["Rate_Lag1"].fillna(df_imputed["Overall_Crude_Rate"])
        
    for cot in cac_cot_dinh_luong:
        if cot in df_imputed.columns and df_imputed[cot].isnull().sum() > 0:
            df_imputed[cot] = df_imputed[cot].fillna(df_imputed[cot].median())
            
    print(f"- Sau xử lý: Tổng số giá trị khuyết còn lại = {df_imputed[cac_cot_dinh_luong].isnull().sum().sum()}")
    return df_imputed

def phat_hien_va_xu_ly_ngoai_lai_3sigma(df, cac_cot_dinh_luong):
    print("\n" + "=" * 85)
    print("TASK 2.4: KIỂM ĐỊNH NGOẠI LAI THEO QUY TẮC 3-SIGMA [Mean - 3*Std, Mean + 3*Std] VÀ IQR")
    print("=" * 85)
    
    df_cleaned = df.copy()
    bang_kiem_dinh = []
    
    for cot in cac_cot_dinh_luong:
        if cot not in df.columns:
            continue
        mu = df[cot].mean()
        sigma = df[cot].std()
        nguong_duoi = max(0.0, mu - 3 * sigma)  # Tỷ lệ bệnh không âm
        nguong_tren = mu + 3 * sigma
        
        ngoai_lai = df[(df[cot] < nguong_duoi) | (df[cot] > nguong_tren)]
        q1 = df[cot].quantile(0.25)
        q3 = df[cot].quantile(0.75)
        iqr = q3 - q1
        ngoai_lai_iqr = df[(df[cot] < max(0.0, q1 - 1.5 * iqr)) | (df[cot] > q3 + 1.5 * iqr)]
        
        bang_kiem_dinh.append({
            "Biến số": cot,
            "Mean": round(mu, 2),
            "Std": round(sigma, 2),
            "Ngưỡng dưới (3s)": round(nguong_duoi, 2),
            "Ngưỡng trên (3s)": round(nguong_tren, 2),
            "Ngoại lai 3-Sigma": len(ngoai_lai),
            "Ngoại lai IQR": len(ngoai_lai_iqr)
        })
        
        # Áp dụng phương pháp Winsorizing / Clipping tại ngưỡng 3-Sigma
        if len(ngoai_lai) > 0:
            df_cleaned[cot] = np.clip(df_cleaned[cot], nguong_duoi, nguong_tren)
            
    df_kq_3sigma = pd.DataFrame(bang_kiem_dinh)
    print(df_kq_3sigma.to_string(index=False))
    return df_cleaned, df_kq_3sigma

def truc_quan_hoa_eda(duong_dan_fig01, duong_dan_fig02):
    print("\n" + "=" * 85)
    print("TASK 2.5: XUẤT BIỂU ĐỒ TRỰC QUAN HÓA EDA DỊCH TỄ HỌC (300 DPI) VÀO figures/")
    print("=" * 85)
    
    # 1. Biểu đồ fig01: Xu hướng thời gian chuỗi kép (Dual-Axis Time Series) của tỷ lệ thô vs ASR World
    if os.path.exists(DATA_YEARLY_PATH):
        df_yearly = pd.read_csv(DATA_YEARLY_PATH)
        df_all = df_yearly[df_yearly["cancer_code"] == 1]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
        
        # Subplot 1: Nam giới (Male)
        df_m = df_all[df_all["sex"] == "Male"].sort_values("year")
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        l1 = ax1.plot(df_m["year"], df_m["crude_rate_per_100k"], "o-", color="#1E88E5", linewidth=2.5, label="Crude Rate (/100k)")
        l2 = ax1_twin.plot(df_m["year"], df_m["asr_world_per_100k"], "s--", color="#D81B60", linewidth=2.2, label="ASR World (/100k)")
        
        ax1.set_title("A. XU HƯỚNG MẮC UNG THƯ Ở NAM GIỚI (2000–2017)", fontsize=12, fontweight="bold", pad=12)
        ax1.set_xlabel("Năm ghi nhận", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Tỷ lệ thô (Crude Rate / 100k)", color="#1E88E5", fontsize=11, fontweight="bold")
        ax1_twin.set_ylabel("Tỷ lệ chuẩn hóa ASR World (/ 100k)", color="#D81B60", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        
        lines_m = l1 + l2
        labels_m = [l.get_label() for l in lines_m]
        ax1.legend(lines_m, labels_m, loc="lower right", frameon=True)
        
        # Subplot 2: Nữ giới (Female)
        df_f = df_all[df_all["sex"] == "Female"].sort_values("year")
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        l3 = ax2.plot(df_f["year"], df_f["crude_rate_per_100k"], "o-", color="#00897B", linewidth=2.5, label="Crude Rate (/100k)")
        l4 = ax2_twin.plot(df_f["year"], df_f["asr_world_per_100k"], "s--", color="#FB8C00", linewidth=2.2, label="ASR World (/100k)")
        
        ax2.set_title("B. XU HƯỚNG MẮC UNG THƯ Ở NỮ GIỚI (2000–2017)", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel("Năm ghi nhận", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Tỷ lệ thô (Crude Rate / 100k)", color="#00897B", fontsize=11, fontweight="bold")
        ax2_twin.set_ylabel("Tỷ lệ chuẩn hóa ASR World (/ 100k)", color="#FB8C00", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        
        lines_f = l3 + l4
        labels_f = [l.get_label() for l in lines_f]
        ax2.legend(lines_f, labels_f, loc="lower right", frameon=True)
        
        fig.tight_layout()
        fig.savefig(duong_dan_fig01, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"- Đã lưu Biểu đồ xu hướng thời gian 2 trục Y -> {duong_dan_fig01}")

    # 2. Biểu đồ fig02: Đường cong tỷ lệ mắc đặc hiệu theo độ tuổi (Age-Specific Incidence Rate)
    if os.path.exists(DATA_AGE_PATH):
        df_age = pd.read_csv(DATA_AGE_PATH)
        # Lọc 5 vị trí ung thư phổ biến nhất: All cancers, Lung, Breast, Prostate, Stomach
        target_cancers = [1, 46, 67, 101, 19]
        df_age_filtered = df_age[df_age["cancer_code"].isin(target_cancers) & (df_age["age_code"] <= 18)].copy()
        
        fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
        colors = {1: "#212121", 46: "#D32F2F", 67: "#E91E63", 101: "#1976D2", 19: "#388E3C"}
        linestyles = {1: "-", 46: "--", 67: "-.", 101: ":", 19: "-"}
        
        for c_code, grp in df_age_filtered.groupby("cancer_code"):
            # Tổng hợp ca và dân số qua 2 giới tính cho mỗi độ tuổi
            age_agg = grp.groupby(["age_code", "age_group"], as_index=False).agg({"cases": "sum", "py": "sum"})
            age_agg["rate"] = (age_agg["cases"] / age_agg["py"]) * 100000.0
            site_name = grp["cancer_site"].iloc[0]
            
            ax.plot(
                age_agg["age_group"],
                age_agg["rate"],
                marker="o",
                linewidth=2.5,
                color=colors.get(c_code, "#333333"),
                linestyle=linestyles.get(c_code, "-"),
                label=f"{site_name} (Mã ICD: {grp['icd10'].iloc[0]})"
            )
            
        ax.set_title("ĐƯỜNG CONG TỶ LỆ MẮC UNG THƯ THEO 18 NHÓM TUỔI (AGE-INCIDENCE CURVES)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Nhóm tuổi (Age Groups)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Tỷ lệ mắc đặc hiệu theo tuổi (Số ca / 100.000 người-năm)", fontsize=11, fontweight="bold")
        ax.set_yscale("log")  # Thang đo logarit biểu thị rõ rệt tốc độ tăng lũy thừa theo tuổi
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", frameon=True, fontsize=10)
        plt.xticks(rotation=45)
        
        fig.tight_layout()
        fig.savefig(duong_dan_fig02, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"- Đã lưu Biểu đồ tháp tuổi và nguy cơ khởi phát -> {duong_dan_fig02}")

def main():
    print("=" * 85)
    print("BƯỚC 2: KHÁM PHÁ DỮ LIỆU (EDA) & LÀM SẠCH CHUYÊN SÂU DỮ LIỆU UNG THƯ")
    print("=" * 85)
    
    assert os.path.exists(DATA_PANEL_PATH), f"Không tìm thấy file: {DATA_PANEL_PATH}"
    df_raw = pd.read_csv(DATA_PANEL_PATH)
    
    cac_cot_dinh_luong = [
        "Total_Cases",
        "Total_Population",
        "Overall_Crude_Rate",
        "Rate_Lag1",
        "Lung_Rate",
        "Colorectal_Rate",
        "Stomach_Rate",
        "Liver_Rate"
    ]
    
    # Task 2.1: Thống kê mô tả
    thong_ke_mo_ta_ban_dau(df_raw, cac_cot_dinh_luong)
    
    # Task 2.2: Trùng lặp
    df_dedup = xu_ly_trung_lap(df_raw)
    
    # Task 2.3: Giá trị khuyết
    df_imputed = xu_ly_gia_tri_khuyet(df_dedup, cac_cot_dinh_luong)
    
    # Task 2.4: Ngoại lai 3-Sigma
    df_cleaned, _ = phat_hien_va_xu_ly_ngoai_lai_3sigma(df_imputed, cac_cot_dinh_luong)
    
    # Task 2.5: Trực quan hóa
    truc_quan_hoa_eda(FIG01_PATH, FIG02_PATH)
    
    # Xuất tệp sạch
    df_cleaned.to_csv(OUTPUT_CLEANED_PATH, index=False, encoding="utf-8")
    print(f"\n- Đã lưu tệp dữ liệu đã làm sạch ({len(df_cleaned):,} bản ghi) -> {OUTPUT_CLEANED_PATH}")
    
    # Task 2.6: Self-test assertions
    print("\n" + "=" * 85)
    print("TASK 2.6: THỰC THI SELF-TEST KIỂM ĐỊNH CHẤT LƯỢNG LÀM SẠCH")
    print("=" * 85)
    assert os.path.exists(OUTPUT_CLEANED_PATH), "Lỗi: File dữ liệu sạch không tồn tại"
    assert os.path.exists(FIG01_PATH), "Lỗi: Không tìm thấy file biểu đồ fig01"
    assert os.path.exists(FIG02_PATH), "Lỗi: Không tìm thấy file biểu đồ fig02"
    assert df_cleaned[cac_cot_dinh_luong].isnull().sum().sum() == 0, "Lỗi: Vẫn còn giá trị NaN trong các cột định lượng"
    assert (df_cleaned["Overall_Crude_Rate"] >= 0).all(), "Lỗi: Phát hiện tỷ lệ Crude Rate âm"
    print("SELF-TEST BƯỚC 2: PASS 100%!")

if __name__ == "__main__":
    main()
