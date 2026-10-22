import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

INPUT_FEATURES_PATH = os.path.join("cleanedData", "cancer_transformed_features.csv")
TRAIN_DATA_PATH = os.path.join("cleanedData", "cancer_train_set.csv")
TEST_DATA_PATH = os.path.join("cleanedData", "cancer_test_set.csv")

def chia_tap_du_lieu_chuoi_thoi_gian(df, split_year=2013):
    print("=" * 85)
    print(f"TASK 4.1: CHIA TẬP DỮ LIỆU CHUỖI THỜI GIAN THEO MỐC NĂM {split_year}")
    print("=" * 85)

    df_train = df[df["year"] <= split_year].copy().reset_index(drop=True)
    df_test = df[df["year"] > split_year].copy().reset_index(drop=True)

    print(f"- Train Set (Giai đoạn {df_train['year'].min()} - {df_train['year'].max()}): {len(df_train)} dòng ({len(df_train)/len(df)*100:.1f}%)")
    print(f"- Test Set  (Giai đoạn {df_test['year'].min()} - {df_test['year'].max()}): {len(df_test)} dòng ({len(df_test)/len(df)*100:.1f}%)")
    
    df_train.to_csv(TRAIN_DATA_PATH, index=False, encoding="utf-8")
    df_test.to_csv(TEST_DATA_PATH, index=False, encoding="utf-8")
    return df_train, df_test

def uoc_luong_mo_hinh_day_du(df_train, cac_bien_doc_lap, bien_phu_thuoc):
    print("\n" + "=" * 85)
    print("TASK 4.2: XÂY DỰNG MÔ HÌNH HỒI QUY ĐẦY ĐỦ (FULL MULTIPLE REGRESSION MODEL)")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    X_train = sm.add_constant(df_train[cac_bien_doc_lap].astype(float))
    model_full = sm.OLS(y_train, X_train).fit()
    print(model_full.summary())
    return model_full

def thuat_toan_backward_elimination(df_train, cac_bien_ung_vien, bien_phu_thuoc, nguong_p=0.05):
    print("\n" + "=" * 85)
    print("TASK 4.3: THỰC THI THUẬT TOÁN LỰA CHỌN BIẾN LÙI (BACKWARD ELIMINATION)")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc]
    bien_hien_tai = list(cac_bien_ung_vien)
    nhat_ky_vong_lap = []
    vong_lap = 1

    while True:
        X_train = sm.add_constant(df_train[bien_hien_tai].astype(float))
        model = sm.OLS(y_train, X_train).fit()
        
        # Bỏ hệ số chặn 'const' khi xét p-value loại trừ
        p_values = model.pvalues.drop("const", errors="ignore")
        max_p_value = p_values.max()
        bien_max_p = p_values.idxmax()

        nhat_ky_vong_lap.append({
            "Vòng lặp": vong_lap,
            "Số biến": len(bien_hien_tai),
            "Biến có p-value cao nhất": bien_max_p,
            "Max p-value": f"{max_p_value:.4f}",
            "R-squared": round(model.rsquared, 4),
            "Adj R-squared": round(model.rsquared_adj, 4),
            "Hành động": f"Loại biến '{bien_max_p}'" if max_p_value > nguong_p else "DỪNG LẠI (TỐI ƯU)"
        })

        if max_p_value > nguong_p:
            bien_hien_tai.remove(bien_max_p)
            vong_lap += 1
        else:
            break

    df_nhat_ky = pd.DataFrame(nhat_ky_vong_lap)
    print(df_nhat_ky.to_string(index=False))

    # Ước lượng mô hình tối ưu cuối cùng
    X_optimal = sm.add_constant(df_train[bien_hien_tai].astype(float))
    model_optimal = sm.OLS(y_train, X_optimal).fit()
    
    print("\n" + "=" * 85)
    print("TASK 4.4: BÁO CÁO KẾT QUẢ MÔ HÌNH HỒI QUY TỐI ƯU (OPTIMAL MODEL SUMMARY)")
    print("=" * 85)
    print(model_optimal.summary())
    return model_optimal, bien_hien_tai

def main():
    print("=" * 85)
    print("BƯỚC 4: XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH OLS & LỰA CHỌN BIẾN STEPWISE")
    print("=" * 85)

    assert os.path.exists(INPUT_FEATURES_PATH), f"Không tìm thấy file: {INPUT_FEATURES_PATH}"
    df = pd.read_csv(INPUT_FEATURES_PATH)

    # Task 4.1: Chia tập train/test
    df_train, df_test = chia_tap_du_lieu_chuoi_thoi_gian(df, split_year=2013)

    # Danh sách các biến độc lập ứng viên
    cac_bien_ung_vien = [
        "Rate_Lag1",
        "Colorectal_Rate",
        "Lung_Rate",
        "Stomach_Rate",
        "Liver_Rate",
        "Time_Index",
        "Log_Population",
        "sex_Male",
        "continent_Asia",
        "continent_Central & South America",
        "continent_Europe",
        "continent_North America",
        "continent_Oceania"
    ]
    # Chỉ giữ các biến có trong df_train
    cac_bien_ung_vien = [b for b in cac_bien_ung_vien if b in df_train.columns]
    bien_phu_thuoc = "Overall_Crude_Rate"

    # Task 4.2: Mô hình đầy đủ
    model_full = uoc_luong_mo_hinh_day_du(df_train, cac_bien_ung_vien, bien_phu_thuoc)

    # Task 4.3 & 4.4: Thuật toán Backward Elimination & Mô hình tối ưu
    model_optimal, cac_bien_toi_uu = thuat_toan_backward_elimination(
        df_train, cac_bien_ung_vien, bien_phu_thuoc, nguong_p=0.05
    )

    # Task 4.5: Self-Test
    print("\n" + "=" * 85)
    print("TASK 4.5: THỰC THI SELF-TEST KIỂM TOÁN MÔ HÌNH HỒI QUY BƯỚC 4")
    print("=" * 85)
    assert os.path.exists(TRAIN_DATA_PATH), "Lỗi: Không tìm thấy train set"
    assert os.path.exists(TEST_DATA_PATH), "Lỗi: Không tìm thấy test set"
    assert model_optimal.rsquared > 0.90, f"Lỗi: R-squared của mô hình quá thấp ({model_optimal.rsquared:.4f})"
    assert (model_optimal.pvalues.drop("const").values < 0.05).all(), "Lỗi: Vẫn còn biến độc lập có p-value >= 0.05 trong mô hình tối ưu"
    print(f"- R-squared mô hình tối ưu: {model_optimal.rsquared:.4f} ({model_optimal.rsquared*100:.2f}%)")
    print(f"- Các biến độc lập được giữ lại: {cac_bien_toi_uu}")
    print("SELF-TEST BƯỚC 4: PASS 100%!")

if __name__ == "__main__":
    main()
