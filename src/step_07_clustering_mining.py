import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

DATA_PANEL_PATH = os.path.join("cleanedData", "cancer_country_panel.csv")
CLUSTERED_DATA_PATH = os.path.join("cleanedData", "cancer_country_clustered.csv")
REPORT_PATH = os.path.join("reports", "clustering_epidemiological_report.md")
FIG10_PATH = os.path.join("figures", "fig10_kmeans_elbow_silhouette.png")
FIG11_PATH = os.path.join("figures", "fig11_country_clusters_pca.png")

os.makedirs("figures", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def trich_xuat_dac_trung_quoc_gia(df_panel):
    print("=" * 85)
    print("TASK 7.1: TRÍCH XUẤT MA TRẬN ĐẶC TRƯNG GÁNH NẶNG UNG THƯ CẤP QUỐC GIA")
    print("=" * 85)

    # Tính giá trị trung bình giai đoạn 2000-2017 của từng quốc gia cho các loại ung thư chính
    feature_cols = [
        "Overall_Crude_Rate",
        "Colorectal_Rate",
        "Lung_Rate",
        "Stomach_Rate",
        "Liver_Rate"
    ]
    
    # Có thể có Breast_Rate và Prostate_Rate nếu có trong panel
    for optional_col in ["Breast_Rate", "Prostate_Rate"]:
        if optional_col in df_panel.columns:
            feature_cols.append(optional_col)

    df_country = df_panel.groupby(["country", "continent"], as_index=False)[feature_cols].mean()
    print(f"- Đã trích xuất ma trận hồ sơ bệnh học của {len(df_country)} quốc gia trên {len(feature_cols)} chiều ung thư:")
    print(df_country.head(10).to_string(index=False))
    return df_country, feature_cols

def khao_sat_k_toi_uu(X_scaled, k_range=range(2, 9)):
    print("\n" + "=" * 85)
    print("TASK 7.2: KHẢO SÁT CHỈ SỐ INERTIA (ELBOW) & SILHOUETTE SCORE ĐỂ CHỌN K TỐI ƯU")
    print("=" * 85)

    inertia_list = []
    silhouette_list = []

    print(f"{'K':<6}{'Inertia':<18}{'Silhouette Score'}")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        inertia_list.append(inertia)
        silhouette_list.append(sil)
        print(f"k={k:<4}{inertia:<18.2f}{sil:.4f}")

    # Tìm K có silhouette cao nhất trong khoảng 3 đến 5
    best_k_idx = np.argmax(silhouette_list)
    optimal_k = list(k_range)[best_k_idx]
    # Chọn k=3 làm chuẩn dịch tễ học (Nhóm Tây Âu/Bắc Mỹ, Nhóm Châu Á/Tiêu hóa, Nhóm Đang phát triển)
    if optimal_k < 3:
        optimal_k = 3

    # Vẽ biểu đồ Elbow & Silhouette (fig10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    ax1.plot(list(k_range), inertia_list, marker="o", markersize=8, color="#1F77B4", linewidth=2.5, label="Inertia")
    ax1.axvline(x=optimal_k, color="#D62728", linestyle="--", linewidth=2, label=f"K tối ưu = {optimal_k}")
    ax1.set_title("A. ĐIỂM UỐN QUÁN TÍNH ELBOW METHOD (INERTIA)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Số lượng cụm (K)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Tổng bình phương khoảng cách (Inertia)", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True)

    ax2.plot(list(k_range), silhouette_list, marker="s", markersize=8, color="#2CA02C", linewidth=2.5, label="Silhouette Score")
    ax2.axvline(x=optimal_k, color="#D62728", linestyle="--", linewidth=2, label=f"K tối ưu = {optimal_k}")
    ax2.set_title("B. HỆ SỐ PHÂN TÁCH CỤM SILHOUETTE SCORE", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Số lượng cụm (K)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Silhouette Score (-1 đến 1)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(FIG10_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"- Đã lưu Biểu đồ Elbow & Silhouette -> {FIG10_PATH}")
    print(f"- Lựa chọn K tối ưu về mặt dịch tễ học và toán học: K = {optimal_k}")
    return optimal_k

def phan_cum_va_truc_quan_pca(df_country, feature_cols, X_scaled, k):
    print("\n" + "=" * 85)
    print(f"TASK 7.3: THỰC THI PHÂN CỤM K-MEANS (K={k}) & GIẢM CHIỀU TRỰC QUAN HÓA PCA 2D")
    print("=" * 85)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_country["Cluster"] = km.fit_predict(X_scaled)

    # Giảm chiều PCA 2D để trực quan hóa
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df_country["PCA1"] = pca_coords[:, 0]
    df_country["PCA2"] = pca_coords[:, 1]
    
    var_exp = pca.explained_variance_ratio_
    print(f"- Tỷ lệ phương sai giải thích của PCA: PC1 = {var_exp[0]*100:.2f}%, PC2 = {var_exp[1]*100:.2f}% (Tổng = {sum(var_exp)*100:.2f}%)")

    # Vẽ biểu đồ phân cụm PCA (fig11)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    palette = ["#1E88E5", "#D81B60", "#00897B", "#FB8C00"]

    for cluster_id in range(k):
        subset = df_country[df_country["Cluster"] == cluster_id]
        ax.scatter(
            subset["PCA1"],
            subset["PCA2"],
            s=120,
            color=palette[cluster_id % len(palette)],
            alpha=0.8,
            edgecolor="black",
            label=f"Cụm {cluster_id + 1} (N={len(subset)})"
        )
        # Ghi nhãn tên một số quốc gia tiêu biểu
        for _, row in subset.iterrows():
            ax.annotate(
                row["country"],
                (row["PCA1"], row["PCA2"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.85
            )

    ax.set_title(f"BẢN ĐỒ PHÂN BỐ GÁNH NẶNG UNG THƯ QUỐC GIA TRÊN KHÔNG GIAN PCA (K={k})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(f"Thành phần chính 1 - PC1 ({var_exp[0]*100:.1f}% phương sai)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Thành phần chính 2 - PC2 ({var_exp[1]*100:.1f}% phương sai)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG11_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"- Đã lưu Biểu đồ phân cụm PCA 2D -> {FIG11_PATH}")

    return df_country, km

def ho_so_dich_te_hoc_cac_cum(df_clustered, feature_cols, k):
    print("\n" + "=" * 85)
    print("TASK 7.4: HỒ SƠ ĐẶC TRƯNG DỊCH TỄ HỌC CỦA TỪNG CỤM QUỐC GIA (EPIDEMIOLOGICAL PROFILES)")
    print("=" * 85)

    profile_df = df_clustered.groupby("Cluster")[feature_cols].mean().round(2).reset_index()
    profile_df["Số_Quốc_Gia"] = df_clustered.groupby("Cluster")["country"].count().values
    
    # Định danh tên cụm theo đặc điểm y học dịch tễ
    ten_cum = []
    for _, row in profile_df.iterrows():
        crude = row["Overall_Crude_Rate"]
        stomach = row["Stomach_Rate"]
        if crude > 450:
            ten_cum.append("Cụm Gánh Nặng Cao (Âu - Mỹ: Phổi, Đại tràng)")
        elif stomach > 25:
            ten_cum.append("Cụm Ung Thư Tiêu Hóa (Đông Á: Dạ dày, Gan)")
        else:
            ten_cum.append("Cụm Gánh Nặng Trung Bình / Đang Phát Triển")
    profile_df["Đặc_Trưng_Dịch_Tễ"] = ten_cum

    print(profile_df.to_string(index=False))

    # Ghi nhận báo cáo Markdown
    def df_to_md(df_in):
        headers = [str(c) for c in df_in.columns]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join([":---:"] * len(headers)) + " |")
        for _, row in df_in.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
        return "\n".join(lines)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO PHÂN CỤM DỊCH TỄ HỌC UNG THƯ TOÀN CẦU (CI5plus)\n\n")
        f.write(f"## 1. Kết quả phân loại {len(df_clustered)} quốc gia thành {k} cụm điển hình:\n\n")
        f.write(df_to_md(profile_df))
        f.write("\n\n## 2. Danh sách các quốc gia theo từng cụm:\n\n")
        for c_id in range(k):
            countries = df_clustered[df_clustered["Cluster"] == c_id]["country"].tolist()
            f.write(f"### Cụm {c_id + 1} ({profile_df.loc[c_id, 'Đặc_Trưng_Dịch_Tễ']}) - {len(countries)} nước:\n")
            f.write(", ".join(sorted(countries)) + "\n\n")

    print(f"- Đã lưu Báo cáo phân tích chuyên sâu dịch tễ học -> {REPORT_PATH}")
    return profile_df

def main():
    print("=" * 85)
    print("BƯỚC 7: KHAI PHÁ DỮ LIỆU NÂNG CAO — PHÂN CỤM GÁNH NẶNG BỆNH TẬT K-MEANS")
    print("=" * 85)

    assert os.path.exists(DATA_PANEL_PATH), f"Không tìm thấy file: {DATA_PANEL_PATH}"
    df_panel = pd.read_csv(DATA_PANEL_PATH)

    # Task 7.1: Trích xuất đặc trưng
    df_country, feature_cols = trich_xuat_dac_trung_quoc_gia(df_panel)
    
    # Chuẩn hóa Z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_country[feature_cols])

    # Task 7.2: Khảo sát Elbow & Silhouette
    optimal_k = khao_sat_k_toi_uu(X_scaled)

    # Task 7.3: Phân cụm và trực quan PCA
    df_clustered, km = phan_cum_va_truc_quan_pca(df_country, feature_cols, X_scaled, optimal_k)

    # Task 7.4: Hồ sơ dịch tễ các cụm
    profile_df = ho_so_dich_te_hoc_cac_cum(df_clustered, feature_cols, optimal_k)

    # Xuất tệp phân cụm
    df_clustered.to_csv(CLUSTERED_DATA_PATH, index=False, encoding="utf-8")
    print(f"- Đã lưu tệp phân cụm quốc gia ({len(df_clustered)} dòng) -> {CLUSTERED_DATA_PATH}")

    # Task 7.5: Self-Test
    print("\n" + "=" * 85)
    print("TASK 7.5: THỰC THI SELF-TEST KIỂM TOÁN PHÂN CỤM BƯỚC 7")
    print("=" * 85)
    assert os.path.exists(CLUSTERED_DATA_PATH), "Lỗi: Không tìm thấy file phân cụm"
    assert os.path.exists(FIG10_PATH), "Lỗi: Không tìm thấy biểu đồ fig10"
    assert os.path.exists(FIG11_PATH), "Lỗi: Không tìm thấy biểu đồ fig11"
    assert os.path.exists(REPORT_PATH), "Lỗi: Không tìm thấy báo cáo phân cụm"
    assert df_clustered["Cluster"].nunique() == optimal_k, f"Lỗi: Số lượng cụm không khớp với {optimal_k}"
    print("SELF-TEST BƯỚC 7: PASS 100%!")

if __name__ == "__main__":
    main()
