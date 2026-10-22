# BÁO CÁO KỸ THUẬT: QUY TRÌNH LÀM SẠCH VÀ TIỀN XỬ LÝ DỮ LIỆU UNG THƯ (CI5plus DETAILED DATABASE)

---

## 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU XỬ LÝ

* **Dự án**: Phân tích dữ liệu dịch tễ học ung thư toàn cầu (`cancerDA`).
* **Nguồn dữ liệu gốc**: Cơ sở dữ liệu **CI5plus Detailed (Phiên bản 2024 v1.1 - IARC / WHO)**.
* **Mục tiêu kỹ thuật**:
  1. **Làm sạch dữ liệu**: Phát hiện và xử lý giá trị thiếu (missing values), bản ghi trùng (duplicates), ép kiểu dữ liệu chuẩn xác (type casting) và loại trừ các giá trị biên không hợp lệ.
  2. **Tiền xử lý & Chuẩn hóa**: Ghép nối các từ điển phân loại giải phẫu bệnh (`cancer_dict.csv`) và cơ quan đăng ký địa lý (`id_dict.csv`), phân lập giai đoạn nghiên cứu khoa học, lọc các nhóm ung thư nguyên phát chính để phòng tránh lỗi đếm trùng lặp (*double-counting*).
  3. **Tổng hợp dữ liệu phục vụ EDA**: Tính toán số ca mắc tổng hợp, tỷ lệ thô (*Crude Rate*), tỷ lệ chuẩn hóa theo tuổi thế giới (*Age-Standardized Rate - ASR World*) và cấu trúc tỷ lệ theo giới/tuổi.
  4. **Xuất bản dữ liệu**: Lưu trữ toàn bộ dữ liệu sạch và các bảng tổng hợp sẵn sàng cho phân tích khám phá (EDA) vào thư mục `cleanedData/`.

---

## 2. KẾT QUẢ KIỂM TOÁN CHẤT LƯỢNG DỮ LIỆU BAN ĐẦU (DATA AUDIT)

Tệp dữ liệu gốc [`CI5plus_Detailed/data.csv`](file:///d:/A_Project/cancerDA/CI5plus_Detailed/data.csv) có kích thước **713,92 MB** với **23.236.620 dòng**. Qua khảo sát streaming theo khối (chunks 2.000.000 dòng), cấu trúc dữ liệu được kiểm toán chi tiết như sau:

| Thuộc tính | Tên cột gốc | Kiểu dữ liệu thô | Số giá trị thiếu (NaN) | Tỷ lệ thiếu (%) | Miền giá trị thực tế |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Mã cơ sở đăng ký** | `id_code` | int64 | 0 | 0.00% | 124 mã duy nhất (khớp 100% với `id_dict.csv`) |
| **Giới tính** | `sex` | int64 | 0 | 0.00% | 1: Nam (11.618.310 dòng), 2: Nữ (11.618.310 dòng) |
| **Mã loại ung thư** | `cancer_code` | int64 | 0 | 0.00% | 1 đến 170 (khớp 100% với `cancer_dict.csv`) |
| **Nhóm tuổi** | `age` | int64 | 0 | 0.00% | 1 đến 19 (18 nhóm tuổi 5 năm + nhóm 19 thiếu tuổi) |
| **Số ca mắc mới** | `cases` | int64 | 0 | 0.00% | Min: 0, Max: 78.432 ca, không có số âm |
| **Dân số phơi nhiễm** | `py` | float64 | 1.222.980 | 5.26% | Min: 0, Max: 41.520.000 người-năm |
| **Năm ghi nhận** | `year` | int64 | 0 | 0.00% | 1953 đến 2017 (65 năm) |

### Đặc thù cấu trúc lưới (Cartesian Product Grid)
Dữ liệu IARC CI5plus được thu thập theo dạng lưới nhân Descartes hoàn chỉnh cho mỗi trung tâm đăng ký:
$$\text{Số bản ghi/năm chuẩn} = 124 \text{ registries} \times 2 \text{ sexes} \times 170 \text{ cancer codes} \times 19 \text{ age groups} = 801.040 \text{ dòng/năm}$$

---

## 3. CÁC PHƯƠNG PHÁP & KĨ THUẬT LÀM SẠCH DỮ LIỆU (DATA CLEANING)

### 3.1. Xử lý giá trị thiếu (Missing Value Handling)
* **Phân tích bản chất dịch tễ**: Toàn bộ **1.222.980 ô trống** tại cột `py` (person-years) xuất hiện duy nhất ở dòng có `age = 19` (nhóm tuổi "Missing / Unknown age"). Về mặt dịch tễ học, các cuộc điều tra dân số quốc gia không thể có nhóm dân số "không rõ tuổi", nhưng các ca bệnh thực tế vẫn có thể không ghi nhận được tuổi lúc chẩn đoán.
* **Kỹ thuật xử lý**: 
  - Không xóa bỏ các dòng `age = 19` vì điều này sẽ làm thất thoát số ca bệnh (`cases`) thực tế.
  - Thực hiện điền giá trị thiếu `py = 0.0` cho `age = 19`. Nhờ đó, phép cộng tổng số ca mắc của mọi độ tuổi luôn chính xác 100%, đồng thời tránh được lỗi phép tính `NaN` trong các hàm tổng hợp. Khi tính tỷ lệ theo từng độ tuổi cụ thể hoặc tính ASR, nhóm `age = 19` được xử lý phân tách đúng chuẩn IARC.

### 3.2. Kiểm định và xử lý bản ghi trùng lặp (Deduplication)
* **Kỹ thuật thực hiện**: Kiểm tra tính duy nhất trên tập khóa chính phức hợp:
  $$\text{Composite Key} = (\texttt{id\_code}, \texttt{sex}, \texttt{cancer\_code}, \texttt{age}, \texttt{year})$$
* **Kết quả**: Số bản ghi trùng lặp nội khối và liên khối = **0**. Tính toàn vẹn cấu trúc dữ liệu đạt mức hoàn hảo.

### 3.3. Tối ưu hóa & Chuẩn hóa kiểu dữ liệu (Type Casting & Memory Efficiency)
Bộ dữ liệu thô ngốn hơn 1.2 GB RAM nếu đọc toàn bộ dạng mặc định của Pandas. Pipeline đã ép kiểu tối ưu về bộ nhớ và chuẩn hóa miền giá trị:
* `cases`: Chuyển từ `float64`/`object` về `int32` (không dấu âm).
* `py`: Chuyển về `float64` độ chính xác cao.
* `sex`, `age`: Chuyển về `int8` (nhu cầu lưu trữ 1 byte).
* `year`, `cancer_code`: Chuyển về `int16` (nhu cầu lưu trữ 2 bytes).

### 3.4. Kiểm định miền giá trị hợp lệ (Domain & Integrity Validation)
* Áp dụng mặt nạ lọc logic (*Boolean Mask Filtering*):
  - $\texttt{cases} \ge 0 \land \texttt{py} \ge 0$ (loại trừ giá trị âm do lỗi nhập liệu).
  - $\texttt{sex} \in \{1, 2\}$ và $\texttt{age} \in [1, 19]$.
  - $\texttt{cancer\_code} \in [1, 170]$.
  - $\texttt{id\_code} \in \text{Set}(\text{id\_dict.csv})$.
* **Kết quả**: 100% bản ghi đều hợp lệ, không có dòng nào vi phạm điều kiện biên.

---

## 4. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

### 4.1. Ghép bảng mã và phân giải từ điển giải phẫu bệnh (`cancer_dict.csv`)
* **Thách thức dịch tễ học (Nguy cơ Double Counting)**: Tệp `cancer_dict.csv` chứa 170 mã với 3 cấp độ phân cấp:
  1. **Mã tổng hợp (Code 1)**: `"All cancers excluding non-melanoma skin"` (tổng số mọi loại ung thư trừ ung thư da không hắc tố).
  2. **Vị trí ung thư nguyên phát chính (Major Cancer Sites, 59 vị trí)**: Các dòng có `histo_label` trống (`""`) và `cancer_code != 1` (ví dụ: Vú, Phổi, Tuyến tiền liệt, Dạ dày, Đại tràng, Gan...).
  3. **Phân loại mô bệnh học chi tiết (Histology Subtypes, 110 mã)**: Các dòng có `histo_label` cụ thể (ví dụ: *Squamous cell carcinoma* của Phổi, *Adenocarcinoma* của Thực quản...).
* **Kỹ thuật xử lý**:
  - Gắn cờ logic `is_major_site = True` và `is_all_cancers = True`.
  - Khi tổng hợp dữ liệu toàn cục, chỉ tính trên các mã nguyên phát chính (`is_major_site == True`) để tránh việc cộng gộp trùng lặp giữa vị trí cơ quan cha và các dạng mô bệnh học con.

### 4.2. Ghép bảng mã và phân giải từ điển địa bàn đăng ký (`id_dict.csv`)
* **Phân giải địa danh & chuẩn hóa tên quốc gia**: Tách trường `id_label` thành `country` và `registry` chuẩn tắc.
  - Chuẩn hóa các trường hợp đặc thù: Canada, Hà Lan (*The Netherlands* $\to$ *Netherlands*), Vương quốc Anh (*UK* $\to$ *United Kingdom*), Hoa Kỳ (*USA* $\to$ *United States*), Thổ Nhĩ Kỳ (*Türkiye* $\to$ *Turkey*).
  - Kết quả: Ánh xạ chuẩn xác **124 cơ sở ghi nhận ung thư** về **44 quốc gia** trên **6 châu lục**.
* **Phân định sắc tộc và phòng ngừa nhân đôi dân số**:
  - Trong `id_dict.csv`, có 108 cơ sở có `ethnic_code == 99` (đại diện cho toàn bộ dân số địa bàn - *Total Population*).
  - Có 16 cơ sở là thống kê chi tiết theo chủng tộc (ví dụ tại Hoa Kỳ SEER: Da trắng, Da đen, Gốc Á).
  - Gắn nhãn `is_total_pop = (ethnic_code == 99)`. Khi tổng hợp thống kê cấp quốc gia và toàn cầu, pipeline chỉ tính trên `is_total_pop == True` nhằm triệt tiêu hoàn toàn sai số nhân đôi dân số.

### 4.3. Xác định giai đoạn nghiên cứu (Study Period: 2000 – 2017)
* **Cơ sở khoa học**:
  - Trước năm 2000, số lượng trung tâm đăng ký của CI5plus còn thưa thớt (1953 chỉ có 12.920 dòng/năm; 1980 có 200.260 dòng/năm).
  - Từ năm **2000 đến 2017** (18 năm liên tục): CI5plus đạt độ bao phủ toàn diện và ổn định tuyệt đối (~801.040 bản ghi/năm), áp dụng thống nhất chuẩn phân loại bệnh tật quốc tế **ICD-10** và quy tắc ghi nhận của IARC/IACR.
* **Quy mô sau lọc giai đoạn**: Từ 23.236.620 dòng giảm xuống **13.966.520 bản ghi** chất lượng cao của thế kỷ 21.

### 4.4. Kỹ thuật tính toán các chỉ số dịch tễ phục vụ EDA
1. **Tỷ lệ mắc thô (Crude Incidence Rate)**:
   $$\text{Crude Rate (trên 100.000 dân)} = \frac{\sum_{i=1}^{19} \text{Cases}_i}{\sum_{i=1}^{18} \text{PY}_i} \times 100{,}000$$
2. **Tỷ lệ chuẩn hóa theo tuổi thế giới (Age-Standardized Incidence Rate - ASR World)**:
   Do cấu trúc dân số giữa các quốc gia và qua các thời kỳ có sự già hóa khác nhau, ASR là tiêu chuẩn vàng dịch tễ của WHO/IARC để so sánh khách quan:
   $$\text{ASR World (trên 100.000 dân)} = \sum_{i=1}^{18} \left( \frac{\text{Cases}_i}{\text{PY}_i} \times \frac{w_i}{100{,}000} \right) \times 100{,}000$$
   với $w_i$ là trọng số dân số chuẩn thế giới Segi-Doll 1966 cho 18 nhóm tuổi (từ 0–4 đến 85+), tổng $\sum_{i=1}^{18} w_i = 100{,}000$.

---

## 5. CẤU TRÚC DỮ LIỆU ĐÃ LƯU TRỮ TRONG `cleanedData/`

Toàn bộ kết quả xử lý đã được ghi nhận vào thư mục [`cleanedData/`](file:///d:/A_Project/cancerDA/cleanedData):

```
cancerDA/
│
├── cleanedData/
│   ├── cleaned_cancer_top_sites_2000_2017.csv   (~183.3 MB)
│   ├── eda_cancer_yearly_summary.csv           (~141.9 KB)
│   ├── eda_continent_country_summary.csv       (~2.24 MB)
│   ├── eda_age_incidence_pattern.csv           (~143.3 KB)
│   └── eda_top_cancers_ranking.csv             (~4.2 KB)
│
├── CI5plus_Detailed/                            (Dữ liệu thô ban đầu)
├── src/                                         (Pipeline xử lý)
└── report.md                                    (Báo cáo kỹ thuật này)
```

### Chi tiết các tệp dữ liệu đã xuất bản:

#### 1. [`cleaned_cancer_top_sites_2000_2017.csv`](file:///d:/A_Project/cancerDA/cleanedData/cleaned_cancer_top_sites_2000_2017.csv)
* **Số dòng**: 1.807.432 dòng | **Kích thước**: 183.29 MB.
* **Mô tả**: Tệp vi mô (microdata) chi tiết của các nhóm ung thư phổ biến nhất thế giới giai đoạn 2000–2017 đã được ghép đầy đủ thông tin châu lục, quốc gia, cơ sở ghi nhận, loại bệnh, mã ICD-10, giới tính, nhóm tuổi, số ca và dân số phơi nhiễm.
* **Các trường**: `year`, `continent`, `country`, `registry`, `id_code`, `ethnic_code`, `is_total_pop`, `sex_label`, `cancer_code`, `cancer_site`, `icd10`, `is_major_site`, `age`, `age_group`, `cases`, `py`.

#### 2. [`eda_cancer_yearly_summary.csv`](file:///d:/A_Project/cancerDA/cleanedData/eda_cancer_yearly_summary.csv)
* **Số dòng**: 2.160 dòng | **Kích thước**: 141.9 KB.
* **Mô tả**: Tổng hợp chuỗi thời gian theo **Năm $\times$ Loại ung thư $\times$ Giới tính**.
* **Các trường**: `year`, `cancer_code`, `cancer_site`, `icd10`, `is_major_site`, `sex`, `cases`, `py`, `crude_rate_per_100k`, `asr_world_per_100k`.
* **Mục đích EDA**: Vẽ biểu đồ xu hướng mắc bệnh (Trend Analysis), kiểm tra xem loại ung thư nào đang tăng/giảm qua từng năm sau khi đã triệt tiêu yếu tố già hóa dân số bằng chỉ số ASR.

#### 3. [`eda_continent_country_summary.csv`](file:///d:/A_Project/cancerDA/cleanedData/eda_continent_country_summary.csv)
* **Số dòng**: 32.956 dòng | **Kích thước**: 2.24 MB.
* **Mô tả**: Tổng hợp theo **Châu lục $\times$ Quốc gia $\times$ Năm $\times$ Loại ung thư $\times$ Giới tính**.
* **Các trường**: `continent`, `country`, `year`, `cancer_code`, `cancer_site`, `icd10`, `sex`, `cases`, `py`, `crude_rate_per_100k`.
* **Mục đích EDA**: So sánh gánh nặng ung thư giữa các châu lục (bản đồ nhiệt Choropleth, so sánh Đông Á, Tây Âu, Bắc Mỹ, v.v.).

#### 4. [`eda_age_incidence_pattern.csv`](file:///d:/A_Project/cancerDA/cleanedData/eda_age_incidence_pattern.csv)
* **Số dòng**: 2.280 dòng | **Kích thước**: 143.3 KB.
* **Mô tả**: Phân bố tỷ lệ mắc đặc hiệu theo tuổi (*Age-Specific Incidence Rate*) cho từng loại ung thư và giới tính trên toàn bộ 18 nhóm tuổi (0–4 đến 85+).
* **Các trường**: `cancer_code`, `cancer_site`, `icd10`, `is_major_site`, `sex`, `age_code`, `age_group`, `cases`, `py`, `age_specific_rate_per_100k`.
* **Mục đích EDA**: Khảo sát đường cong nguy cơ theo độ tuổi (Age-Incidence Curve), xác định độ tuổi khởi phát và đỉnh điểm của từng bệnh.

#### 5. [`eda_top_cancers_ranking.csv`](file:///d:/A_Project/cancerDA/cleanedData/eda_top_cancers_ranking.csv)
* **Số dòng**: 60 dòng | **Kích thước**: 4.2 KB.
* **Mô tả**: Bảng tổng kết xếp hạng toàn cầu của tất cả 59 vị trí ung thư nguyên phát chính trong suốt 18 năm (2000–2017).

---

## 6. PHÁT HIỆN DỊCH TỄ HỌC SƠ BỘ TỪ KẾT QUẢ TIỀN XỬ LÝ

Bảng xếp hạng Top 15 loại ung thư có số ca mắc lớn nhất thế giới trong giai đoạn 2000–2017 trích xuất từ [`eda_top_cancers_ranking.csv`](file:///d:/A_Project/cancerDA/cleanedData/eda_top_cancers_ranking.csv):

| Xếp hạng | Vị trí Ung thư | Mã ICD-10 | Tổng số ca mắc (2000-2017) | Tỷ lệ trong tổng số ung thư (%) | Số ca Nam | Số ca Nữ | Tỷ lệ Nữ (%) | Tỷ lệ Nam/Nữ | Tỷ lệ thô (/100k) | ASR Thế giới (/100k) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Ung thư Vú (Breast)** | C50 | **7.786.189** | 13.52% | 63.666 | 7.722.523 | 99.2% | 0.01 | 60.83 | **40.56** |
| **2** | **Ung thư Phổi (Lung)** | C34 | **7.498.276** | 13.02% | 4.380.071 | 3.118.205 | 41.6% | 1.40 | 58.58 | **34.26** |
| **3** | **Ung thư Tuyến tiền liệt (Prostate)** | C61 | **7.074.994** | 12.29% | 7.074.994 | 0 | 0.0% | N/A | 55.27 | **33.99** |
| **4** | **Ung thư Đại tràng (Colon)** | C18 | **4.272.090** | 7.42% | 2.177.130 | 2.094.960 | 49.0% | 1.04 | 33.38 | **19.03** |
| **5** | **Ung thư Bàng quang (Bladder)** | C67 | **2.373.888** | 4.12% | 1.797.960 | 575.928 | 24.3% | 3.12 | 18.55 | **10.24** |
| **6** | **U hắc tố da (Melanoma of skin)** | C43 | **2.191.491** | 3.81% | 1.205.980 | 985.511 | 45.0% | 1.22 | 17.12 | **11.38** |
| **7** | **Ung thư Trực tràng (Rectum)** | C19-20 | **2.100.379** | 3.65% | 1.251.314 | 849.065 | 40.4% | 1.47 | 16.41 | **10.15** |
| **8** | **Ung thư Dạ dày (Stomach)** | C16 | **2.060.264** | 3.58% | 1.320.801 | 739.463 | 35.9% | 1.79 | 16.10 | **9.78** |
| **9** | **U lympho không Hodgkin (NHL)** | C82-86,C96 | **2.018.125** | 3.50% | 1.102.668 | 915.457 | 45.4% | 1.20 | 15.77 | **10.04** |
| **10** | **Ung thư Tuyến giáp (Thyroid)** | C73 | **1.815.154** | 3.15% | 400.732 | 1.414.422 | 77.9% | 0.28 | 14.18 | **11.11** |
| **11** | **Ung thư Thận (Kidney)** | C64 | **1.675.721** | 2.91% | 1.057.498 | 618.223 | 36.9% | 1.71 | 13.09 | **8.60** |
| **12** | **Ung thư Tụy (Pancreas)** | C25 | **1.564.439** | 2.72% | 796.356 | 768.083 | 49.1% | 1.04 | 12.22 | **6.93** |
| **13** | **Ung thư Thân tử cung (Corpus uteri)**| C54 | **1.472.770** | 2.56% | 0 | 1.472.770 | 100.0%| 0.00 | 11.51 | **7.59** |
| **14** | **Ung thư Gan (Liver)** | C22 | **1.399.368** | 2.43% | 999.420 | 399.948 | 28.6% | 2.50 | 10.93 | **6.99** |
| **15** | **Ung thư Buồng trứng (Ovary)** | C56 | **879.407** | 1.53% | 0 | 879.407 | 100.0%| 0.00 | 6.87 | **4.51** |

### Các điểm nhấn dịch tễ đáng chú ý:
1. **Bộ ba ung thư hàng đầu**: Ung thư Vú, Ung thư Phổi và Ung thư Tuyến tiền liệt chiếm tổng cộng **38,83%** toàn bộ số ca ung thư mới trên thế giới trong giai đoạn nghiên cứu.
2. **Khác biệt giới tính sâu sắc**:
   - Tỷ lệ Nam/Nữ ở Ung thư Bàng quang là **3,12 : 1**; Ung thư Gan là **2,50 : 1**; Ung thư Phổi là **1,40 : 1**.
   - Ngược lại, Ung thư Tuyến giáp có **77,9%** số ca mắc xảy ra ở Nữ giới (Tỷ lệ Nam/Nữ chỉ là **0,28 : 1**).
3. **Đại trực tràng tổng hợp (Colorectum = Colon + Rectum)**: Nếu gộp cả Đại tràng (4,27 triệu ca) và Trực tràng (2,10 triệu ca), ung thư Đại trực tràng có tổng cộng **6,37 triệu ca mắc**, đứng vị trí thứ 4 toàn cầu và là bệnh ung thư đường tiêu hóa phổ biến nhất.

---

## 7. KẾT LUẬN & HƯỚNG DẪN SỬ DỤNG CHO BƯỚC TIẾP THEO

Toàn bộ quy trình tiền xử lý đã hoàn thành một cách tự động, chính xác và có thể tái lập (*100% reproducible*) thông qua mã nguồn [`src/clean_and_preprocess.py`](file:///d:/A_Project/cancerDA/src/clean_and_preprocess.py).

Các tệp CSV trong thư mục [`cleanedData/`](file:///d:/A_Project/cancerDA/cleanedData) đã được chuẩn hóa hoàn hảo để bạn có thể:
1. Đọc trực tiếp vào Pandas / Polars chỉ mất vài mili-giây.
2. Vẽ các biểu đồ EDA (Seaborn, Matplotlib, Plotly) về xu thế theo năm, bản đồ theo quốc gia, tháp tuổi bệnh tật.
3. Sử dụng ngay làm đầu vào cho các thuật toán học máy, hồi quy chuỗi thời gian hoặc phân cụm (Clustering) sau này.
