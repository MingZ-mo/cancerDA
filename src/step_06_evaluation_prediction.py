import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

TRAIN_DATA_PATH = os.path.join("cleanedData", "cancer_train_set.csv")
TEST_DATA_PATH = os.path.join("cleanedData", "cancer_test_set.csv")
FIG08_PATH = os.path.join("figures", "fig08_model_comparison_forecast.png")
FIG09_PATH = os.path.join("figures", "fig09_test_actual_vs_predicted.png")

OPTIMAL_FEATURES = [
    "Rate_Lag1",
    "Colorectal_Rate",
    "Lung_Rate",
    "Stomach_Rate",
    "Time_Index",
    "Log_Population",
    "sex_Male",
    "continent_Europe",
    "continent_North America",
    "continent_Oceania"
]
TARGET = "Overall_Crude_Rate"

def tinh_toan_cac_chi_so_goc_da_sgu(y_true, y_pred, n_obs, k_features, llf):
    ess = np.sum((y_true - y_pred) ** 2)
    y_bar = np.mean(y_true)
    tss = np.sum((y_true - y_bar) ** 2)
    r2 = 1.0 - (ess / tss)
    adj_r2 = 1.0 - ((ess / (n_obs - k_features - 1)) / (tss / (n_obs - 1)))
    aic = -2.0 * llf + 2.0 * (k_features + 1)
    bic = -2.0 * llf + np.log(n_obs) * (k_features + 1)
    return {"ESS": ess, "TSS": tss, "R2": r2, "Adj_R2": adj_r2, "AIC": aic, "BIC": bic}

def huan_luyen_va_so_sanh_mo_hinh(df_train, df_test, bien_phu_thuoc, features):
    print("=" * 85)
    print("TASK 6.1: TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH THEO CÔNG THỨC GỐC DA-SGU")
    print("=" * 85)

    y_train = df_train[bien_phu_thuoc].values
    y_test = df_test[bien_phu_thuoc].values

    # 1. Mô hình OLS Tuyến tính Chuẩn & Robust HC3
    X_train_mat = sm.add_constant(df_train[features].astype(float))
    X_test_mat = sm.add_constant(df_test[features].astype(float))
    
    m_ols = sm.OLS(y_train, X_train_mat).fit()
    m_hc3 = sm.OLS(y_train, X_train_mat).fit(cov_type="HC3")

    calc_ols = tinh_toan_cac_chi_so_goc_da_sgu(
        y_train, m_ols.fittedvalues, len(y_train), len(features), m_ols.llf
    )
    print(f"- ESS: {calc_ols['ESS']:,.2f} | TSS: {calc_ols['TSS']:,.2f}")
    print(f"- R2 : {calc_ols['R2']:.4f} | Adj R2: {calc_ols['Adj_R2']:.4f} | AIC: {calc_ols['AIC']:.2f} | BIC: {calc_ols['BIC']:.2f}")

    # 2. Dự báo trên tập kiểm thử Test Set (2014-2017)
    pred_test_ols = m_ols.predict(X_test_mat)
    rmse_ols = np.sqrt(mean_squared_error(y_test, pred_test_ols))
    mae_ols = mean_absolute_error(y_test, pred_test_ols)
    mape_ols = mean_absolute_percentage_error(y_test, pred_test_ols) * 100.0

    # 3. Mô hình Bán Logarit (Log-Linear Model: ln(Y) ~ X)
    y_train_log = np.log(y_train)
    m_log = sm.OLS(y_train_log, X_train_mat).fit()
    pred_test_log = np.exp(m_log.predict(X_test_mat))
    rmse_log = np.sqrt(mean_squared_error(y_test, pred_test_log))
    mae_log = mean_absolute_error(y_test, pred_test_log)
    mape_log = mean_absolute_percentage_error(y_test, pred_test_log) * 100.0

    print("\n" + "=" * 85)
    print("TASK 6.2: SO SÁNH HIỆU NĂNG MÔ HÌNH TRÊN TẬP KIỂM THỬ NGOÀI MẪU (TEST SET 2014–2017)")
    print("=" * 85)
    df_compare = pd.DataFrame([
        {
            "Mô hình": "1. OLS Tuyến tính (Linear OLS)",
            "Train R²": f"{m_ols.rsquared:.4f}",
            "AIC": f"{m_ols.aic:.1f}",
            "BIC": f"{m_ols.bic:.1f}",
            "Test RMSE": f"{rmse_ols:.2f}",
            "Test MAE": f"{mae_ols:.2f}",
            "Test MAPE (%)": f"{mape_ols:.2f}%"
        },
        {
            "Mô hình": "2. OLS Robust HC3 (Kháng dị phương sai)",
            "Train R²": f"{m_hc3.rsquared:.4f}",
            "AIC": f"{m_hc3.aic:.1f}",
            "BIC": f"{m_hc3.bic:.1f}",
            "Test RMSE": f"{rmse_ols:.2f}",
            "Test MAE": f"{mae_ols:.2f}",
            "Test MAPE (%)": f"{mape_ols:.2f}%"
        },
        {
            "Mô hình": "3. Log-Linear Model (ln(Y) ~ X)",
            "Train R²": f"{m_log.rsquared:.4f}",
            "AIC": f"{m_log.aic:.1f}",
            "BIC": f"{m_log.bic:.1f}",
            "Test RMSE": f"{rmse_log:.2f}",
            "Test MAE": f"{mae_log:.2f}",
            "Test MAPE (%)": f"{mape_log:.2f}%"
        }
    ])
    print(df_compare.to_string(index=False))

    return m_ols, m_hc3, pred_test_ols, pred_test_log, {
        "rmse_ols": rmse_ols,
        "mae_ols": mae_ols,
        "mape_ols": mape_ols
    }

def truc_quan_hoa_danh_gia_du_bao(df_test, y_test, pred_test_ols, fig08_path, fig09_path):
    print("\n" + "=" * 85)
    print("TASK 6.3: XUẤT BIỂU ĐỒ ĐÁNH GIÁ DỰ BÁO TRỰC QUAN (300 DPI) VÀO figures/")
    print("=" * 85)

    # 1. Biểu đồ fig08: Chuỗi thời gian trung bình tỷ lệ mắc Test vs Dự báo theo từng năm 2014-2017
    df_eval = pd.DataFrame({
        "year": df_test["year"].values,
        "Actual": y_test,
        "Predicted": pred_test_ols
    })

    yearly_eval = df_eval.groupby("year")[["Actual", "Predicted"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(yearly_eval["year"], yearly_eval["Actual"], "o-", color="#1E88E5", linewidth=2.5, markersize=8, label="Giá trị Thực tế (Test Actual)")
    ax.plot(yearly_eval["year"], yearly_eval["Predicted"], "s--", color="#D32F2F", linewidth=2.2, markersize=8, label="Giá trị Dự báo OLS (Forecast)")
    
    ax.set_title("ĐÁNH GIÁ DỰ BÁO NGOÀI MẪU: THỰC TẾ VS DỰ BÁO (2014–2017)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Năm ghi nhận (Giai đoạn kiểm thử ngoài mẫu)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tỷ lệ mắc thô trung bình (/100.000 dân)", fontsize=11, fontweight="bold")
    ax.set_xticks(yearly_eval["year"])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(fig08_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"- Đã lưu Biểu đồ dự báo ngoài mẫu -> {fig08_path}")

    # 2. Biểu đồ fig09: Scatter Test Actual vs Predicted với dải tin cậy
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.scatter(y_test, pred_test_ols, color="#00897B", alpha=0.6, edgecolors="black", s=45, label="Quan sát kiểm thử")

    min_v = min(y_test.min(), pred_test_ols.min())
    max_v = max(y_test.max(), pred_test_ols.max())
    ax.plot([min_v, max_v], [min_v, max_v], color="#D32F2F", linestyle="--", linewidth=2.5, label="Đường lý tưởng (y = x)")

    ax.set_title("BIỂU ĐỒ ĐỐI CHIẾU: THỰC TẾ VS DỰ BÁO TRÊN TEST SET", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Tỷ lệ mắc ung thư Thực tế (/100k dân)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tỷ lệ mắc ung thư Dự báo (/100k dân)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(fig09_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"- Đã lưu Biểu đồ đối chiếu phân tán -> {fig09_path}")

def main():
    print("=" * 85)
    print("BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH HỒI QUY, SO SÁNH HIỆU NĂNG & DỰ BÁO NGOÀI MẪU")
    print("=" * 85)

    assert os.path.exists(TRAIN_DATA_PATH), f"Không tìm thấy file: {TRAIN_DATA_PATH}"
    assert os.path.exists(TEST_DATA_PATH), f"Không tìm thấy file: {TEST_DATA_PATH}"

    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    m_ols, m_hc3, pred_test_ols, pred_test_log, metrics = huan_luyen_va_so_sanh_mo_hinh(
        df_train, df_test, TARGET, OPTIMAL_FEATURES
    )

    truc_quan_hoa_danh_gia_du_bao(
        df_test, df_test[TARGET].values, pred_test_ols, FIG08_PATH, FIG09_PATH
    )

    # Self-test
    print("\n" + "=" * 85)
    print("TASK 6.4: THỰC THI SELF-TEST KIỂM TOÁN CHỈ SỐ ĐÁNH GIÁ BƯỚC 6")
    print("=" * 85)
    assert os.path.exists(FIG08_PATH), "Lỗi: Không tìm thấy fig08"
    assert os.path.exists(FIG09_PATH), "Lỗi: Không tìm thấy fig09"
    assert metrics["mape_ols"] < 10.0, f"Lỗi: Sai số MAPE quá lớn ({metrics['mape_ols']:.2f}%)"
    print(f"- Độ chính xác ngoài mẫu: MAPE = {metrics['mape_ols']:.2f}% (Độ chính xác {100 - metrics['mape_ols']:.2f}%)")
    print("SELF-TEST BƯỚC 6: PASS 100%!")

if __name__ == "__main__":
    main()
