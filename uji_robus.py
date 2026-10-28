import cv2
import numpy as np
import os
import glob
import pandas as pd

def evaluate_feature_matching_pair(img1, img2, ratio_thresh=0.75, ransac_thresh=5.0):
    """
    Evaluasi kuantitatif pasangan citra menggunakan SIFT + FLANN + RANSAC.
    Mengembalikan: (kp1_count, kp2_count, total_matches, inliers_count, outliers_count, precision_ratio)
    """
    if img1 is None or img2 is None:
        return 0, 0, 0, 0, 0, 0.0

    # Konversi ke Grayscale jika citra berwarna/3-channel
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    # 1. Ekstraksi Fitur SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    kp1_count = len(kp1) if kp1 else 0
    kp2_count = len(kp2) if kp2 else 0

    if des1 is None or des2 is None or kp1_count < 4 or kp2_count < 4:
        return kp1_count, kp2_count, 0, 0, 0, 0.0

    # 2. Matching dengan FLANN Matcher & Lowe's Ratio Test
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return kp1_count, kp2_count, 0, 0, 0, 0.0

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    total_matches = len(good_matches)
    if total_matches < 4:
        return kp1_count, kp2_count, total_matches, 0, total_matches, 0.0

    # 3. Estimasi Homografi RANSAC (Filter Inliers vs Outliers)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if mask is not None:
        inliers_count = int(np.sum(mask))
        outliers_count = total_matches - inliers_count
    else:
        inliers_count = 0
        outliers_count = total_matches

    # 4. Hitung Precision Ratio (%)
    precision_ratio = (inliers_count / total_matches * 100.0) if total_matches > 0 else 0.0

    return kp1_count, kp2_count, total_matches, inliers_count, outliers_count, precision_ratio


def run_robustness_evaluation(base_path, scenarios):
    all_details = []

    print("=" * 120)
    print(f"{'Skenario':<10} | {'Metode':<10} | {'Pasangan Frame':<22} | {'Avg KP':<8} | {'Matches':<8} | {'Inliers':<8} | {'Precision (%)':<12}")
    print("=" * 120)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', 
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')

    for scn in scenarios:
        methods_folders = {
            'Original': os.path.join(base_path, scn, "input"),
            'GHE': os.path.join(base_path, scn, "output_ghe"),
            'CLAHE': os.path.join(base_path, scn, "output_clahe"),
            'LogCh': os.path.join(base_path, scn, "output_logch")
        }

        # Mengambil daftar berkas citra dari folder Original
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(methods_folders['Original'], ext)))
        image_files = sorted(list(set(image_files)))

        if len(image_files) < 2:
            print(f"[{scn.upper()}] Peringatan: Butuh minimal 2 berkas citra di folder input.")
            continue

        # Evaluasi Pasangan Frame Berurutan (Frame_i vs Frame_i+1)
        for i in range(len(image_files) - 1):
            file1 = os.path.basename(image_files[i])
            file2 = os.path.basename(image_files[i + 1])
            pair_label = f"{file1} - {file2}"

            for method_name, folder_path in methods_folders.items():
                p1 = os.path.join(folder_path, file1)
                p2 = os.path.join(folder_path, file2)

                if os.path.exists(p1) and os.path.exists(p2):
                    img1 = cv2.imread(p1)
                    img2 = cv2.imread(p2)
                    
                    kp1_cnt, kp2_cnt, total_m, inliers, outliers, precision = evaluate_feature_matching_pair(img1, img2)
                    avg_kp = round((kp1_cnt + kp2_cnt) / 2.0, 1)
                    
                    print(f"{scn:<10} | {method_name:<10} | {pair_label:<22} | {avg_kp:<8} | {total_m:<8} | {inliers:<8} | {precision:<12.2f}")

                    all_details.append({
                        'Skenario': scn,
                        'Metode': method_name,
                        'Pasangan_Frame': pair_label,
                        'Avg_Keypoints': avg_kp,
                        'Total_Matches': total_m,
                        'Valid_Matches_Inliers': inliers,
                        'Outliers': outliers,
                        'Precision_Ratio (%)': round(precision, 2)
                    })

    print("=" * 120)
    return all_details

# --- Eksekusi Program Utama ---

base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"]

# Jalankan Evaluasi Robustness Ekstraksi dan Pencocokan Fitur
evaluation_data = run_robustness_evaluation(base_path, scenarios)

# --- Pengolahan Data & Ekspor Hasil ke Berkas CSV ---
if evaluation_data:
    df_detail = pd.DataFrame(evaluation_data)
    
    # 1. Simpan Data Rinci per Pasangan Citra
    csv_detail_path = os.path.join(base_path, "evaluasi_robustness_fitur_detail.csv")
    df_detail.to_csv(csv_detail_path, index=False)

    # 2. Hitung Rata-Rata Metrik per Skenario dan Metode untuk Tabel Pembahasan Paper
    df_summary = df_detail.groupby(['Skenario', 'Metode'])[[
        'Avg_Keypoints', 
        'Total_Matches', 
        'Valid_Matches_Inliers', 
        'Outliers', 
        'Precision_Ratio (%)'
    ]].mean().round(2).reset_index()

    csv_summary_path = os.path.join(base_path, "evaluasi_robustness_fitur_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)

    print("\n" + "=" * 80)
    print("   RINGKASAN EVALUASI ROBUSTNESS FITUR GEOMETRIS (RATA-RATA PER SKENARIO)   ")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    print(f"\n[SUKSES] Rekap data rinci disimpan ke   : {csv_detail_path}")
    print(f"[SUKSES] Ringkasan rata-rata disimpan ke : {csv_summary_path}")