import cv2
import numpy as np
import os
import glob
import pandas as pd

def compute_ransac_inliers(img1, img2, ratio_thresh=0.75, ransac_thresh=5.0):
    """
    Menghitung jumlah Valid Matches (Inliers) menggunakan SIFT + FLANN + RANSAC.
    """
    if img1 is None or img2 is None:
        return 0, 0, 0  # Total Matches, Inliers, Outliers

    # Konversi ke Grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    # 1. Ekstraksi SIFT Keypoints & Descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0, 0, 0

    # 2. Matching dengan FLANN Matcher & Lowe's Ratio Test
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0, 0, 0

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    total_matches = len(good_matches)
    if total_matches < 4:
        return total_matches, 0, total_matches

    # 3. Estimasi Homografi menggunakan RANSAC untuk Mendapatkan Inliers
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if mask is not None:
        inliers_count = int(np.sum(mask))
        outliers_count = total_matches - inliers_count
    else:
        inliers_count = 0
        outliers_count = total_matches

    return total_matches, inliers_count, outliers_count


def process_ransac_matching_analysis(base_path, scenarios):
    all_results = []

    print("=" * 105)
    print(f"{'Skenario':<10} | {'Pasangan Citra':<25} | {'Original':<10} | {'GHE':<10} | {'CLAHE':<10} | {'Log-Ch':<10}")
    print("=" * 105)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', 
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')

    for scn in scenarios:
        folders = {
            'Original': os.path.join(base_path, scn, "input"),
            'GHE': os.path.join(base_path, scn, "output_ghe"),
            'CLAHE': os.path.join(base_path, scn, "output_clahe"),
            'LogCh': os.path.join(base_path, scn, "output_logch")
        }

        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folders['Original'], ext)))
        image_files = sorted(list(set(image_files)))

        if len(image_files) < 2:
            print(f"[{scn.upper()}] Peringatan: Butuh minimal 2 gambar berurutan di '{folders['Original']}'")
            continue

        # Evaluasi Pasangan Frame Berurutan (Frame_i vs Frame_i+1)
        for i in range(len(image_files) - 1):
            file1 = os.path.basename(image_files[i])
            file2 = os.path.basename(image_files[i + 1])
            pair_name = f"{file1} - {file2}"

            inliers_dict = {}

            for method_name, folder_path in folders.items():
                p1 = os.path.join(folder_path, file1)
                p2 = os.path.join(folder_path, file2)

                if os.path.exists(p1) and os.path.exists(p2):
                    img1 = cv2.imread(p1)
                    img2 = cv2.imread(p2)
                    _, inliers, _ = compute_ransac_inliers(img1, img2)
                    inliers_dict[method_name] = inliers
                else:
                    inliers_dict[method_name] = 0

            print(f"{scn:<10} | {pair_name:<25} | {inliers_dict['Original']:<10} | "
                  f"{inliers_dict['GHE']:<10} | {inliers_dict['CLAHE']:<10} | {inliers_dict['LogCh']:<10}")

            all_results.append({
                'Skenario': scn,
                'Pasangan_Citra': pair_name,
                'Valid_Matches_Original': inliers_dict['Original'],
                'Valid_Matches_GHE': inliers_dict['GHE'],
                'Valid_Matches_CLAHE': inliers_dict['CLAHE'],
                'Valid_Matches_LogCh': inliers_dict['LogCh']
            })

    print("=" * 105)
    return all_results

# --- Eksekusi Program ---

base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"]

# Jalankan analisis RANSAC Valid Matches
data_ransac = process_ransac_matching_analysis(base_path, scenarios)

# --- Olah Data dan Simpan ke CSV ---
if data_ransac:
    df = pd.DataFrame(data_ransac)
    
    # 1. Simpan Data Detail per Pasangan Frame
    csv_detail_path = os.path.join(base_path, "komparasi_valid_matches_ransac_detail.csv")
    df.to_csv(csv_detail_path, index=False)
    
    # 2. Hitung Rata-Rata Valid Matches per Skenario untuk Tabel Paper
    df_summary = df.groupby('Skenario')[[
        'Valid_Matches_Original', 
        'Valid_Matches_GHE', 
        'Valid_Matches_CLAHE', 
        'Valid_Matches_LogCh'
    ]].mean().reset_index()
    
    csv_summary_path = os.path.join(base_path, "komparasi_valid_matches_ransac_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)

    print("\n--- RATA-RATA VALID MATCHES (RANSAC INLIERS) PER SKENARIO ---")
    print(df_summary.to_string(index=False))
    print(f"\n[SUKSES] Rekap data detail disimpan ke   : {csv_detail_path}")
    print(f"[SUKSES] Ringkasan rata-rata disimpan ke : {csv_summary_path}")
