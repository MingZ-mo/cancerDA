import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

INPUT_CLEANED_PATH = os.path.join("cleanedData", "cancer_country_panel_cleaned.csv")
OUTPUT_FEATURES_PATH = os.path.join("cleanedData", "cancer_transformed_features.csv")
FIG03_PATH = os.path.join("figures", "fig03_correlation_matrix.png")

def thuc_hien_roi_rac_hoa_va_label_encoding(df):
    print("=" * 85)
    print("TASK 3.1: RỜI RẠC HÓA DỮ LIỆU DỊCH TỄ (pd.cut) & MÃ HÓA NHÃN (LabelEncoder)")
    print("=" * 85)

    df_trans = df.copy()
    nhan_nguy_co = ["Thap", "Trung_Binh", "Cao"]
    # Phân loại mức độ gánh nặng ung thư (Crude Rate Risk Level)
    df_trans["Risk_Category"] = pd.cut(
        df_trans["Overall_Crude_Rate"], 
        bins=3, 
        labels=nhan_nguy_co
    )
    
    le = LabelEncoder()
    df_trans["Risk_Category_Encoded"] = le.fit_transform(df_trans["Risk_Category"].astype(str))
    
    print("- Bảng phân bổ các mức độ nguy cơ ung thư:")
    print(df_trans["Risk_Category"].value_counts())
    print(f"- Ánh xạ LabelEncoder: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df_trans, le

def thuc_hien_one_hot_encoding(df):
    print("\n" + "=" * 85)
    print("TASK 3.2: MÃ HÓA BIẾN ĐỊNH DANH BẰNG ONE-HOT ENCODING (pd.get_dummies)")
    print("=" * 85)

    df_encoded = df.copy()
    # Mã hóa One-Hot cho Châu lục và Giới tính, áp dụng drop_first=True phòng ngừa bẫy biến giả (Dummy Trap)
    cac_bien_danh_muc = ["continent", "sex"]
    df_encoded = pd.get_dummies(df_encoded, columns=cac_bien_danh_muc, drop_first=True, dtype=int)
    
    cot_dummy_moi = [col for col in df_encoded.columns if any(col.startswith(p + "_") for p in cac_bien_danh_muc)]
    print(f"- Các cột biến giả (Dummies) được sinh ra ({len(cot_dummy_moi)} cột):")
    for c in cot_dummy_moi:
        print(f"  + {c}")
    return df_encoded

def phan_tich_tuong_quan_pearson(df, cac_bien_so, bien_muc_tieu):
    print("\n" + "=" * 85)
    print(f"TASK 3.3: PHÂN TÍCH TƯƠNG QUAN TUYẾN TÍNH PEARSON VỚI '{bien_muc_tieu}'")
    print("=" * 85)

    bang_tuong_quan = []
    for bien in cac_bien_so:
        if bien not in df.columns or bien == bien_muc_tieu:
            continue
        corr, p_value = pearsonr(df[bien], df[bien_muc_tieu])
        
        # Mức độ tương quan
        abs_r = abs(corr)
        if abs_r >= 0.7:
            muc_do = "Rất mạnh (Strong)"
        elif abs_r >= 0.5:
            muc_do = "Mạnh (Moderate)"
        elif abs_r >= 0.3:
            muc_do = "Trung bình"
        else:
            muc_do = "Yếu (Weak)"
            
        y_nghia = "Có ý nghĩa (p < 0.05)" if p_value < 0.05 else "Không có ý nghĩa"
        
        bang_tuong_quan.append({
            "Biến độc lập": bien,
            "Hệ số r": round(corr, 4),
            "p-value": f"{p_value:.4e}",
            "Mức độ tương quan": muc_do,
            "Ý nghĩa thống kê": y_nghia
        })
        
    df_corr = pd.DataFrame(bang_tuong_quan).sort_values(by="Hệ số r", ascending=False)
    print(df_corr.to_string(index=False))
    return df_corr

def truc_quan_hoa_heatmap_tuong_quan(df, cac_bien_phan_tich, duong_dan_fig03):
    print("\n" + "=" * 85)
    print("TASK 3.4: XUẤT MA TRẬN NHIỆT TƯƠNG QUAN (CORRELATION HEATMAP 300 DPI) figures/")
    print("=" * 85)

    df_subset = df[cac_bien_phan_tich].copy()
    corr_matrix = df_subset.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        linewidths=0.8, 
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("MA TRẬN HỆ SỐ TƯƠNG QUAN PEARSON GIỮA CÁC BIẾN DỊCH TỄ HỌC", fontsize=12, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    fig.tight_layout()
    fig.savefig(duong_dan_fig03, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"- Đã lưu heatmap tương quan -> {duong_dan_fig03}")

def main():
    print("=" * 85)
    print("BƯỚC 3: BIẾN ĐỔI BIẾN SỐ, ĐẶC TRƯNG & PHÂN TÍCH TƯƠNG QUAN PEARSON")
    print("=" * 85)

    assert os.path.exists(INPUT_CLEANED_PATH), f"Không tìm thấy file: {INPUT_CLEANED_PATH}"
    df = pd.read_csv(INPUT_CLEANED_PATH)

    # Task 3.1: Rời rạc hóa & Label Encoding
    df_step3_1, le = thuc_hien_roi_rac_hoa_va_label_encoding(df)

    # Task 3.2: One-Hot Encoding
    df_step3_2 = thuc_hien_one_hot_encoding(df_step3_1)

    # Task 3.3: Phân tích tương quan
    cac_bien_dinh_luong_mo_hinh = [
        "Rate_Lag1",
        "Colorectal_Rate",
        "Lung_Rate",
        "Stomach_Rate",
        "Liver_Rate",
        "Time_Index",
        "Log_Population"
    ]
    df_corr_result = phan_tich_tuong_quan_pearson(
        df_step3_2, 
        cac_bien_dinh_luong_mo_hinh, 
        bien_muc_tieu="Overall_Crude_Rate"
    )

    # Task 3.4: Heatmap
    cac_bien_heatmap = ["Overall_Crude_Rate"] + cac_bien_dinh_luong_mo_hinh
    truc_quan_hoa_heatmap_tuong_quan(df_step3_2, cac_bien_heatmap, FIG03_PATH)

    # Xuất tệp đã biến đổi
    df_step3_2.to_csv(OUTPUT_FEATURES_PATH, index=False, encoding="utf-8")
    print(f"\n- Đã lưu tệp đặc trưng mô hình ({len(df_step3_2):,} dòng, {len(df_step3_2.columns)} cột) -> {OUTPUT_FEATURES_PATH}")

    # Task 3.5: Self-Test Assertions
    print("\n" + "=" * 85)
    print("TASK 3.5: THỰC THI SELF-TEST KIỂM ĐỊNH KẾT QUẢ BIẾN ĐỔI")
    print("=" * 85)
    assert os.path.exists(OUTPUT_FEATURES_PATH), "Lỗi: Không tìm thấy file features đã biến đổi"
    assert os.path.exists(FIG03_PATH), "Lỗi: Không tìm thấy file heatmap fig03"
    assert "Risk_Category_Encoded" in df_step3_2.columns, "Lỗi: Thiếu cột Risk_Category_Encoded"
    assert "sex_Male" in df_step3_2.columns, "Lỗi: Thiếu cột dummy sex_Male"
    assert df_step3_2["Risk_Category_Encoded"].nunique() == 3, "Lỗi: Label Encoding phải có 3 mức phân loại"
    print("SELF-TEST BƯỚC 3: PASS 100%!")

if __name__ == "__main__":
    main()
