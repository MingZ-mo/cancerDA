import os
import sys
import time
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# --- 1. CONFIGURATION & DIRECTORIES ---
RAW_DATA_PATH = "CI5plus_Detailed/data.csv"
CANCER_DICT_PATH = "CI5plus_Detailed/cancer_dict.csv"
ID_DICT_PATH = "CI5plus_Detailed/id_dict.csv"
OUTPUT_DIR = "cleanedData"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Study period: 2000 - 2017 (Modern 21st century era with highest registry completeness)
START_YEAR = 2000
END_YEAR = 2017

# Segi World Standard Population (Weights per 100,000 for age groups 1 to 18)
SEGI_WORLD_WEIGHTS = {
    1: 12000,   # 0-4
    2: 10000,   # 5-9
    3: 9000,    # 10-14
    4: 9000,    # 15-19
    5: 8000,    # 20-24
    6: 8000,    # 25-29
    7: 6000,    # 30-34
    8: 6000,    # 35-39
    9: 6000,    # 40-44
    10: 6000,   # 45-49
    11: 5000,   # 50-54
    12: 4000,   # 55-59
    13: 4000,   # 60-64
    14: 3000,   # 65-69
    15: 2000,   # 70-74
    16: 1000,   # 75-79
    17: 500,    # 80-84
    18: 500     # 85+
}

AGE_LABEL_MAP = {
    1: "0-4",
    2: "5-9",
    3: "10-14",
    4: "15-19",
    5: "20-24",
    6: "25-29",
    7: "30-34",
    8: "35-39",
    9: "40-44",
    10: "45-49",
    11: "50-54",
    12: "55-59",
    13: "60-64",
    14: "65-69",
    15: "70-74",
    16: "75-79",
    17: "80-84",
    18: "85+",
    19: "Missing"
}

CONTINENT_MAP = {
    1: "Africa",
    2: "Central & South America",
    3: "North America",
    4: "Asia",
    5: "Europe",
    6: "Oceania"
}

# Top representative major cancers for focused datasets & ranking
TOP_CANCER_CODES = [
    1,   # All cancers excluding non-melanoma skin
    11,  # Oesophagus
    19,  # Stomach
    21,  # Colon
    22,  # Rectum
    31,  # Liver
    42,  # Pancreas
    46,  # Lung
    63,  # Melanoma of skin
    67,  # Breast
    70,  # Cervix uteri
    78,  # Corpus uteri
    86,  # Ovary
    101, # Prostate
    109, # Kidney
    112, # Bladder
    132, # Brain, CNS
    142, # Thyroid
    154, # Hodgkin lymphoma
    160, # Non-Hodgkin lymphoma
    164, # Lymphoid leukaemia
    167  # Myeloid leukaemia
]


def load_and_preprocess_dictionaries():
    print("[BƯỚC 1] Đang tải và chuẩn hóa các bảng mã từ điển...")
    
    # 1. Cancer dictionary
    df_cancer = pd.read_csv(CANCER_DICT_PATH, encoding="utf-8")
    df_cancer['histo_label'] = df_cancer['histo_label'].fillna("")
    # Flag major site: cancer_code != 1 and histo_label is blank
    df_cancer['is_major_site'] = (df_cancer['histo_label'] == "") & (df_cancer['cancer_code'] != 1)
    df_cancer['is_all_cancers'] = (df_cancer['cancer_code'] == 1)
    df_cancer['cancer_site'] = df_cancer['cancer_label']
    
    # 2. Registry ID dictionary
    df_id = pd.read_csv(ID_DICT_PATH, encoding="utf-8")
    df_id['continent'] = df_id['CI5_continent'].map(CONTINENT_MAP)
    
    # Parse country and registry clean names
    def parse_label(label):
        if "Canada" in label:
            return "Canada", label
        if "T\ufffd\ufffdrkiye" in label or "Türkiye" in label:
            parts = label.split(",")
            return "Turkey", parts[1].strip() if len(parts) > 1 else "Turkey"
        if "The Netherlands" in label:
            return "Netherlands", "Netherlands"
        if "UK," in label:
            parts = label.split(",")
            return "United Kingdom", parts[1].strip()
        if "USA," in label:
            parts = label.split(",")
            return "United States", ", ".join(parts[1:]).strip()
        if ":" in label:
            country = label.split(":")[0].strip()
            registry = label.split(":")[1].strip()
            return country, registry
        parts = label.split(",")
        if len(parts) == 1:
            return parts[0].strip(), parts[0].strip()
        else:
            return parts[0].strip(), ", ".join(parts[1:]).strip()

    countries = []
    registries = []
    for lbl in df_id['id_label']:
        c, r = parse_label(lbl)
        countries.append(c)
        registries.append(r)
        
    df_id['country'] = countries
    df_id['registry'] = registries
    df_id['is_total_pop'] = (df_id['ethnic_code'] == 99)
    
    print(f"  + Cancer dictionary: {len(df_cancer)} mã ({df_cancer['is_major_site'].sum()} major sites, 1 mã all-cancers, {len(df_cancer) - df_cancer['is_major_site'].sum() - 1} mô học).")
    print(f"  + Registry dictionary: {len(df_id)} cơ sở đăng ký tại {df_id['country'].nunique()} quốc gia thuộc {df_id['continent'].nunique()} châu lục.")
    
    return df_cancer, df_id


def process_pipeline():
    start_time = time.time()
    df_cancer, df_id = load_and_preprocess_dictionaries()
    
    # Map lookups for fast processing
    cancer_meta = df_cancer.set_index('cancer_code')[['cancer_site', 'icd10', 'is_major_site', 'is_all_cancers']].to_dict('index')
    id_meta = df_id.set_index('id_code')[['continent', 'country', 'registry', 'ethnic_code', 'is_total_pop']].to_dict('index')
    
    # Major cancer codes: code 1 (All cancers) + all 60 major sites
    major_and_all_codes = set(df_cancer[df_cancer['is_major_site'] | df_cancer['is_all_cancers']]['cancer_code'])
    top_codes_set = set(TOP_CANCER_CODES)
    
    print("\n[BƯỚC 2] Bắt đầu duyệt và làm sạch dữ liệu lớn (Stream Processing CI5plus)...")
    print(f"  - Giai đoạn nghiên cứu đã chọn: {START_YEAR} – {END_YEAR}")
    print(f"  - Kích thước khối (chunk size): 2,000,000 dòng")

    # Accumulators for EDA tables
    # 1. Yearly summary: (year, cancer_code, sex) -> {cases, py, age_cases: {age: cases}, age_py: {age: py}}
    yearly_acc = {}
    
    # 2. Continent/Country summary: (continent, country, year, cancer_code, sex) -> {cases, py}
    geo_acc = {}
    
    # 3. Age incidence summary: (cancer_code, sex, age) -> {cases, py}
    age_acc = {}
    
    # 4. Overall ranking summary: (cancer_code) -> {cases_male, cases_female, py_male, py_female, age_cases_total: {age: cases}, age_py_total: {age: py}}
    ranking_acc = {}

    # File writers for cleaned records
    cleaned_top_records = []
    
    total_raw_rows = 0
    total_in_period = 0
    total_filtered_major = 0
    total_invalid_filtered = 0

    chunk_idx = 0
    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=2_000_000):
        chunk_idx += 1
        total_raw_rows += len(chunk)
        
        # --- CLEANING STEP A: FILTER STUDY PERIOD ---
        chunk = chunk[(chunk['year'] >= START_YEAR) & (chunk['year'] <= END_YEAR)].copy()
        if chunk.empty:
            continue
        total_in_period += len(chunk)
        
        # --- CLEANING STEP B: TYPE CASTING & VALIDATION ---
        # 1. Check cases: must be numeric, >= 0
        chunk['cases'] = pd.to_numeric(chunk['cases'], errors='coerce')
        # 2. Check py: numeric, >= 0, missing py for age 19 -> fill with 0.0
        chunk['py'] = pd.to_numeric(chunk['py'], errors='coerce')
        chunk['py'] = chunk['py'].fillna(0.0)
        
        # Validate domains
        valid_mask = (
            (chunk['cases'] >= 0) & 
            (chunk['py'] >= 0) & 
            (chunk['sex'].isin([1, 2])) & 
            (chunk['age'].between(1, 19)) &
            (chunk['cancer_code'].between(1, 170)) &
            (chunk['id_code'].isin(id_meta))
        )
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            total_invalid_filtered += invalid_count
            chunk = chunk[valid_mask].copy()

        chunk['cases'] = chunk['cases'].astype(np.int32)
        chunk['py'] = chunk['py'].astype(np.float64)
        chunk['sex'] = chunk['sex'].astype(np.int8)
        chunk['age'] = chunk['age'].astype(np.int8)
        chunk['year'] = chunk['year'].astype(np.int16)
        chunk['cancer_code'] = chunk['cancer_code'].astype(np.int16)
        
        # --- PREPROCESSING STEP C: ENRICH METADATA ---
        # Add id dictionary fields
        chunk['is_total_pop'] = chunk['id_code'].map(lambda x: id_meta[x]['is_total_pop'])
        
        # Filter for major cancer sites to avoid double counting
        chunk_major = chunk[chunk['cancer_code'].isin(major_and_all_codes)].copy()
        total_filtered_major += len(chunk_major)

        # Collect subset for cleaned top cancers csv
        chunk_top = chunk_major[chunk_major['cancer_code'].isin(top_codes_set)].copy()
        if not chunk_top.empty:
            chunk_top['continent'] = chunk_top['id_code'].map(lambda x: id_meta[x]['continent'])
            chunk_top['country'] = chunk_top['id_code'].map(lambda x: id_meta[x]['country'])
            chunk_top['registry'] = chunk_top['id_code'].map(lambda x: id_meta[x]['registry'])
            chunk_top['ethnic_code'] = chunk_top['id_code'].map(lambda x: id_meta[x]['ethnic_code'])
            chunk_top['cancer_site'] = chunk_top['cancer_code'].map(lambda x: cancer_meta[x]['cancer_site'])
            chunk_top['icd10'] = chunk_top['cancer_code'].map(lambda x: cancer_meta[x]['icd10'])
            chunk_top['is_major_site'] = chunk_top['cancer_code'].map(lambda x: cancer_meta[x]['is_major_site'])
            chunk_top['sex_label'] = chunk_top['sex'].map({1: "Male", 2: "Female"})
            chunk_top['age_group'] = chunk_top['age'].map(AGE_LABEL_MAP)
            
            cleaned_top_records.append(chunk_top[[
                'year', 'continent', 'country', 'registry', 'id_code', 'ethnic_code', 'is_total_pop',
                'sex_label', 'cancer_code', 'cancer_site', 'icd10', 'is_major_site',
                'age', 'age_group', 'cases', 'py'
            ]])

        # --- PREPROCESSING STEP D: ACCUMULATE EDA METRICS ---
        # Use only total populations (ethnic_code == 99) to avoid duplicating regional populations
        chunk_eda = chunk_major[chunk_major['is_total_pop']].copy()
        
        if not chunk_eda.empty:
            # 1. Yearly accumulator
            for (y, cc, s), grp in chunk_eda.groupby(['year', 'cancer_code', 'sex']):
                k = (y, cc, s)
                if k not in yearly_acc:
                    yearly_acc[k] = {'cases': 0, 'py': 0.0, 'age_c': {}, 'age_p': {}}
                yearly_acc[k]['cases'] += grp['cases'].sum()
                yearly_acc[k]['py'] += grp['py'].sum()
                for age_val, sub in grp.groupby('age'):
                    yearly_acc[k]['age_c'][age_val] = yearly_acc[k]['age_c'].get(age_val, 0) + sub['cases'].sum()
                    yearly_acc[k]['age_p'][age_val] = yearly_acc[k]['age_p'].get(age_val, 0.0) + sub['py'].sum()
                    
            # 2. Geo accumulator (Top codes only to keep file size optimal and focused)
            chunk_geo = chunk_eda[chunk_eda['cancer_code'].isin(top_codes_set)].copy()
            if not chunk_geo.empty:
                chunk_geo['continent'] = chunk_geo['id_code'].map(lambda x: id_meta[x]['continent'])
                chunk_geo['country'] = chunk_geo['id_code'].map(lambda x: id_meta[x]['country'])
                for (cont, ctry, y, cc, s), grp in chunk_geo.groupby(['continent', 'country', 'year', 'cancer_code', 'sex']):
                    gk = (cont, ctry, y, cc, s)
                    if gk not in geo_acc:
                        geo_acc[gk] = {'cases': 0, 'py': 0.0}
                    geo_acc[gk]['cases'] += grp['cases'].sum()
                    geo_acc[gk]['py'] += grp['py'].sum()
                    
            # 3. Age accumulator
            for (cc, s, a), grp in chunk_eda.groupby(['cancer_code', 'sex', 'age']):
                ak = (cc, s, a)
                if ak not in age_acc:
                    age_acc[ak] = {'cases': 0, 'py': 0.0}
                age_acc[ak]['cases'] += grp['cases'].sum()
                age_acc[ak]['py'] += grp['py'].sum()
                
            # 4. Overall ranking accumulator (all 60 major sites + code 1)
            for (cc, s), grp in chunk_eda.groupby(['cancer_code', 'sex']):
                if cc not in ranking_acc:
                    ranking_acc[cc] = {
                        'cases_male': 0, 'cases_female': 0,
                        'py_male': 0.0, 'py_female': 0.0,
                        'age_c': {}, 'age_p': {}
                    }
                c_sum = grp['cases'].sum()
                p_sum = grp['py'].sum()
                if s == 1:
                    ranking_acc[cc]['cases_male'] += c_sum
                    ranking_acc[cc]['py_male'] += p_sum
                else:
                    ranking_acc[cc]['cases_female'] += c_sum
                    ranking_acc[cc]['py_female'] += p_sum
                for age_val, sub in grp.groupby('age'):
                    ranking_acc[cc]['age_c'][age_val] = ranking_acc[cc]['age_c'].get(age_val, 0) + sub['cases'].sum()
                    ranking_acc[cc]['age_p'][age_val] = ranking_acc[cc]['age_p'].get(age_val, 0.0) + sub['py'].sum()

        print(f"  + Hoàn tất khối {chunk_idx}: Duyệt qua {total_raw_rows:,} dòng thô.")

    print(f"\n[BƯỚC 3] Hoàn thành duyệt dữ liệu!")
    print(f"  - Tổng số bản ghi thô: {total_raw_rows:,}")
    print(f"  - Số bản ghi trong giai đoạn 2000-2017: {total_in_period:,}")
    print(f"  - Số bản ghi Major Sites & All Cancers: {total_filtered_major:,}")
    print(f"  - Số bản ghi không hợp lệ bị loại bỏ: {total_invalid_filtered:,}")

    # --- SAVE 1: CLEANED RECORDS CSV ---
    print("\n[BƯỚC 4] Đang lưu trữ các tệp kết quả vào folder 'cleanedData/'...")
    print("  1. Xuất 'cleanedData/cleaned_cancer_top_sites_2000_2017.csv'...")
    df_cleaned_top = pd.concat(cleaned_top_records, ignore_index=True)
    cleaned_top_csv_path = os.path.join(OUTPUT_DIR, "cleaned_cancer_top_sites_2000_2017.csv")
    df_cleaned_top.to_csv(cleaned_top_csv_path, index=False, encoding="utf-8-sig")
    print(f"     => Đã ghi {len(df_cleaned_top):,} dòng | Dung lượng: {os.path.getsize(cleaned_top_csv_path) / (1024*1024):.2f} MB")

    # --- SAVE 2: YEARLY SUMMARY (CRUDE & ASR) ---
    print("  2. Xuất 'cleanedData/eda_cancer_yearly_summary.csv'...")
    yearly_rows = []
    for (y, cc, s), val in yearly_acc.items():
        total_c = val['cases']
        total_p = val['py']
        crude_rate = (total_c / total_p * 100000.0) if total_p > 0 else 0.0
        
        # Calculate ASR World Standard
        asr = 0.0
        for age_code in range(1, 19):
            ac = val['age_c'].get(age_code, 0)
            ap = val['age_p'].get(age_code, 0.0)
            w = SEGI_WORLD_WEIGHTS.get(age_code, 0)
            if ap > 0:
                asr += (ac / ap) * (w / 100000.0) * 100000.0
                
        meta = cancer_meta[cc]
        yearly_rows.append({
            'year': y,
            'cancer_code': cc,
            'cancer_site': meta['cancer_site'],
            'icd10': meta['icd10'],
            'is_major_site': meta['is_major_site'],
            'sex': "Male" if s == 1 else "Female",
            'cases': total_c,
            'py': np.round(total_p, 1),
            'crude_rate_per_100k': np.round(crude_rate, 2),
            'asr_world_per_100k': np.round(asr, 2)
        })
    df_yearly = pd.DataFrame(yearly_rows).sort_values(by=['year', 'cancer_code', 'sex'])
    yearly_csv_path = os.path.join(OUTPUT_DIR, "eda_cancer_yearly_summary.csv")
    df_yearly.to_csv(yearly_csv_path, index=False, encoding="utf-8-sig")
    print(f"     => Đã ghi {len(df_yearly):,} dòng | Dung lượng: {os.path.getsize(yearly_csv_path) / (1024*1024):.2f} MB")

    # --- SAVE 3: GEOGRAPHIC SUMMARY ---
    print("  3. Xuất 'cleanedData/eda_continent_country_summary.csv'...")
    geo_rows = []
    for (cont, ctry, y, cc, s), val in geo_acc.items():
        total_c = val['cases']
        total_p = val['py']
        crude_rate = (total_c / total_p * 100000.0) if total_p > 0 else 0.0
        meta = cancer_meta[cc]
        geo_rows.append({
            'continent': cont,
            'country': ctry,
            'year': y,
            'cancer_code': cc,
            'cancer_site': meta['cancer_site'],
            'icd10': meta['icd10'],
            'sex': "Male" if s == 1 else "Female",
            'cases': total_c,
            'py': np.round(total_p, 1),
            'crude_rate_per_100k': np.round(crude_rate, 2)
        })
    df_geo = pd.DataFrame(geo_rows).sort_values(by=['continent', 'country', 'year', 'cancer_code', 'sex'])
    geo_csv_path = os.path.join(OUTPUT_DIR, "eda_continent_country_summary.csv")
    df_geo.to_csv(geo_csv_path, index=False, encoding="utf-8-sig")
    print(f"     => Đã ghi {len(df_geo):,} dòng | Dung lượng: {os.path.getsize(geo_csv_path) / (1024*1024):.2f} MB")

    # --- SAVE 4: AGE DISTRIBUTION ---
    print("  4. Xuất 'cleanedData/eda_age_incidence_pattern.csv'...")
    age_rows = []
    for (cc, s, a), val in age_acc.items():
        total_c = val['cases']
        total_p = val['py']
        rate = (total_c / total_p * 100000.0) if total_p > 0 else 0.0
        meta = cancer_meta[cc]
        age_rows.append({
            'cancer_code': cc,
            'cancer_site': meta['cancer_site'],
            'icd10': meta['icd10'],
            'is_major_site': meta['is_major_site'],
            'sex': "Male" if s == 1 else "Female",
            'age_code': a,
            'age_group': AGE_LABEL_MAP[a],
            'cases': total_c,
            'py': np.round(total_p, 1),
            'age_specific_rate_per_100k': np.round(rate, 2)
        })
    df_age = pd.DataFrame(age_rows).sort_values(by=['cancer_code', 'sex', 'age_code'])
    age_csv_path = os.path.join(OUTPUT_DIR, "eda_age_incidence_pattern.csv")
    df_age.to_csv(age_csv_path, index=False, encoding="utf-8-sig")
    print(f"     => Đã ghi {len(df_age):,} dòng | Dung lượng: {os.path.getsize(age_csv_path) / (1024*1024):.2f} MB")

    # --- SAVE 5: GLOBAL CANCER RANKING & COMPARISON ---
    print("  5. Xuất 'cleanedData/eda_top_cancers_ranking.csv'...")
    all_cancers_cases = ranking_acc.get(1, {}).get('cases_male', 0) + ranking_acc.get(1, {}).get('cases_female', 0)
    
    ranking_rows = []
    for cc, val in ranking_acc.items():
        if cc == 1:
            continue # Don't rank "All Cancers" against individual cancer sites
        meta = cancer_meta[cc]
        if not meta['is_major_site']:
            continue
            
        cm = val['cases_male']
        cf = val['cases_female']
        total_c = cm + cf
        total_p = val['py_male'] + val['py_female']
        
        crude_rate = (total_c / total_p * 100000.0) if total_p > 0 else 0.0
        pct_of_all = (total_c / all_cancers_cases * 100.0) if all_cancers_cases > 0 else 0.0
        sex_ratio = (cm / cf) if cf > 0 else np.nan
        
        # Calculate overall ASR
        asr = 0.0
        for age_code in range(1, 19):
            ac = val['age_c'].get(age_code, 0)
            ap = val['age_p'].get(age_code, 0.0)
            w = SEGI_WORLD_WEIGHTS.get(age_code, 0)
            if ap > 0:
                asr += (ac / ap) * (w / 100000.0) * 100000.0
                
        ranking_rows.append({
            'cancer_code': cc,
            'cancer_site': meta['cancer_site'],
            'icd10': meta['icd10'],
            'total_cases': total_c,
            'cases_male': cm,
            'cases_female': cf,
            'pct_female': np.round(cf / total_c * 100.0, 1) if total_c > 0 else 0.0,
            'sex_ratio_m_f': np.round(sex_ratio, 2),
            'pct_of_all_cancers': np.round(pct_of_all, 2),
            'crude_rate_per_100k': np.round(crude_rate, 2),
            'asr_world_per_100k': np.round(asr, 2)
        })
        
    df_ranking = pd.DataFrame(ranking_rows).sort_values(by='total_cases', ascending=False).reset_index(drop=True)
    df_ranking.index = df_ranking.index + 1
    df_ranking.index.name = 'global_rank'
    df_ranking = df_ranking.reset_index()
    
    ranking_csv_path = os.path.join(OUTPUT_DIR, "eda_top_cancers_ranking.csv")
    df_ranking.to_csv(ranking_csv_path, index=False, encoding="utf-8-sig")
    print(f"     => Đã ghi {len(df_ranking):,} dòng | Dung lượng: {os.path.getsize(ranking_csv_path) / (1024*1024):.2f} MB")

    elapsed = time.time() - start_time
    print(f"\n[HOÀN TẤT] Toàn bộ pipeline làm sạch và tiền xử lý đã hoàn thành trong {elapsed:.2f} giây!")
    print(f"Các tệp đã được lưu an toàn tại thư mục '{OUTPUT_DIR}/'.")

    return df_ranking.head(15)

if __name__ == "__main__":
    top15 = process_pipeline()
    print("\n--- TOP 15 UNG THƯ HÀNG ĐẦU THẾ GIỚI (2000 - 2017) ---")
    print(top15[['global_rank', 'cancer_site', 'icd10', 'total_cases', 'pct_of_all_cancers', 'asr_world_per_100k']].to_string(index=False))
