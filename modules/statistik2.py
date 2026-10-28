import cv2
import numpy as np
import pandas as pd
import os
import glob

def compute_pair_feature_stats(img1, img2, detector_type='sift'):
    """
    Menghitung statistik pencocokan fitur antara dua bingkai berurutan.
    """
    if detector_type.lower() == 'sift':
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=3000)
        norm_type = cv2.NORM_HAMMING

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    n_kp1 = len(kp1) if kp1 is not None else 0
    n_kp2 = len(kp2) if kp2 is not None else 0
    total_kp = n_kp1 + n_kp2
    avg_kp = total_kp / 2.0

    if des1 is None or des2 is None or n_kp1 < 4 or n_kp2 < 4:
        return {
            'KP_Frame1': n_kp1, 'KP_Frame2': n_kp2, 'Total_KP': total_kp,
            'Good_Matches': 0, 'Inliers': 0, 'Precision_%': 0.0, 'Efficiency_%': 0.0
        }

    # Matcher + Lowe's Ratio Test
    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    n_good = len(good_matches)
    n_inliers = 0

    # RANSAC Filtering
    if n_good >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            n_inliers = int(np.sum(mask))

    # Rasio Presisi (Inliers vs Good Matches) & Efisiensi Fitur (Inliers vs Avg KP)
    precision = (n_inliers / n_good * 100.0) if n_good > 0 else 0.0
    efficiency = (n_inliers / avg_kp * 100.0) if avg_kp > 0 else 0.0

    return {
        'KP_Frame1': n_kp1,
        'KP_Frame2': n_kp2,
        'Total_KP': total_kp,
        'Good_Matches': n_good,
        'Inliers': n_inliers,
        'Precision_%': round(precision, 2),
        'Efficiency_%': round(efficiency, 2)
    }

def generate_detailed_feature_report(base_path, detector_type='sift'):
    scenarios = ["cctv", "medic", "underwater"]
    methods = {
        "Raw (Input)": "input",
        "Global HE": "output_ghe",
        "CLAHE": "output_clahe",
        "Proposed Log-Ch": "output_logch"
    }

    raw_records = []

    for scn in scenarios:
        for method_label, folder_name in methods.items():
            folder_path = os.path.join(base_path, scn, folder_name)
            
            exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', 
                    '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(folder_path, ext)))
            files = sorted(list(set(files)))

            if len(files) < 2:
                continue

            for i in range(len(files) - 1):
                img1 = cv2.imread(files[i])
                img2 = cv2.imread(files[i+1])
                
                if img1 is None or img2 is None:
                    continue

                stats = compute_pair_feature_stats(img1, img2, detector_type=detector_type)
                stats['Scenario'] = scn.upper()
                stats['Method'] = method_label
                stats['Pair'] = f"Frame_{i}->{i+1}"
                raw_records.append(stats)

    df = pd.DataFrame(raw_records)
    
    if df.empty:
        print("❌ Tidak ada data gambar yang berhasil diproses.")
        return

    # 1. TAMPILKAN SUMMARY AGREGAT PER METODE & SKENARIO
    summary_df = df.groupby(['Scenario', 'Method']).agg(
        Total_KP_Mean=('Total_KP', 'mean'),
        Good_Matches_Mean=('Good_Matches', 'mean'),
        Inliers_Mean=('Inliers', 'mean'),
        Precision_Mean=('Precision_%', 'mean'),
        Precision_Std=('Precision_%', 'std'),
        Efficiency_Mean=('Efficiency_%', 'mean')
    ).reset_index()

    # Format Tampilan DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    print("\n" + "="*90)
    print(f"📊 RINGKASAN STATISTIK EVALUASI FITUR ({detector_type.upper()})")
    print("="*90)
    print(summary_df.to_string(index=False))

    # 2. SIMPAN HASIL KE CSV UNTUK TABEL PAPER
    output_csv = "statistik_fitur_detail.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Laporan lengkap per-pasangan bingkai disimpan ke: '{output_csv}'")

# --- EKSEKUSI PROGRAM ---
base_directory = "data_penelitian"
generate_detailed_feature_report(base_directory, detector_type='sift')
