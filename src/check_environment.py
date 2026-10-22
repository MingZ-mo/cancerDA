import sys
import os
import platform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def kiem_tra_moi_truong():
    print("=" * 65)
    print("BẮT ĐẦU KIỂM TRA MÔI TRƯỜNG PHÂN TÍCH DỮ LIỆU (DA-SGU)")
    print("=" * 65)

    python_version = sys.version.replace('\n', ' ')
    print(f"[THÔNG TIN] Phiên bản Python: {python_version}")
    print(f"[THÔNG TIN] Đường dẫn thực thi Python: {sys.executable}")
    print(f"[THÔNG TIN] Hệ điều hành: {platform.system()} {platform.release()} ({platform.architecture()[0]})")

    assert sys.version_info >= (3, 9), "Lỗi: Phiên bản Python phải từ 3.9 trở lên!"

    cac_thu_vien_bat_buoc = [
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "sklearn",
        "matplotlib",
        "seaborn",
        "openpyxl"
    ]

    ket_qua_kiem_tra = {}

    for ten_thu_vien in cac_thu_vien_bat_buoc:
        try:
            module = __import__(ten_thu_vien)
            phien_ban = getattr(module, "__version__", "Không xác định")
            ket_qua_kiem_tra[ten_thu_vien] = {
                "trang_thai": "PASS",
                "phien_ban": phien_ban
            }
            print(f"[PASS] Thư viện '{ten_thu_vien:<12}' | Trạng thái: SẴN SÀNG | Phiên bản: {phien_ban}")
        except ImportError as e:
            ket_qua_kiem_tra[ten_thu_vien] = {
                "trang_thai": "FAIL",
                "loi": str(e)
            }
            print(f"[FAIL] Thư viện '{ten_thu_vien:<12}' | CHƯA được cài đặt! Lỗi: {e}")

    cac_thu_vien_loi = [k for k, v in ket_qua_kiem_tra.items() if v["trang_thai"] == "FAIL"]
    assert len(cac_thu_vien_loi) == 0, f"Self-Test Thất bại! Các thư viện chưa cài: {cac_thu_vien_loi}"

    print("-" * 65)
    print("SELF-TEST THÀNH CÔNG: Mọi thư viện bắt buộc đã được cài đặt và kiểm thử 100%!")
    print("=" * 65)
    return ket_qua_kiem_tra

if __name__ == "__main__":
    kiem_tra_moi_truong()
