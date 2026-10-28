import cv2
import numpy as np
import os
import glob
import pandas as pd
from skimage.metrics import structural_similarity as ssim

def evaluate_registration_metrics(img1, img2, ratio_thresh=0.75, ransac_thresh=5.0):
    """
    Evaluasi Metrik Registrasi dan Pencocokan Fitur:
    - SIFT + FLANN + RANSAC (Matches, Inliers, Precision Ratio, RMSE)
    - Image Warping & Alignment (SSIM)
    """
    if img1 is None or img2 is None:
        return 0, 0, 0, 0, 0.0, 999.0, 0.0

    # Konversi ke Grayscale jika berwarna
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    # 1. Ekstraksi Fitur SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    kp1_cnt = len(kp1) if kp1 else 0
    kp2_cnt = len(kp2) if kp2 else 0

    if des1 is None or des2 is None or kp1_cnt < 4 or kp2_cnt < 4:
        return kp1_cnt, kp2_cnt, 0, 0, 0.0, 999.0, 0.0

    # 2. Matching dengan FLANN Matcher & Ratio Test
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return kp1_cnt, kp2_cnt, 0, 0, 0.0, 999.0, 0.0

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    total_matches = len(good_matches)
    if total_matches < 4:
        return kp1_cnt, kp2_cnt, total_matches, 0, 0.0, 999.0, 0.0

    # 3. Estimasi Homografi RANSAC & Perhitungan RMSE Geometris
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if mask is not None and H is not None:
        inliers_mask = mask.ravel() == 1
        inliers_cnt = int(np.sum(inliers_mask))
        
        # Hitung RMSE Eror Proyeksi Geometris Piksel pada Inliers
        if inliers_cnt > 0:
            src_inliers = src_pts[inliers_mask]
            dst_inliers = dst_pts[inliers_mask]
            
            # Proyeksi titik src menggunakan Homografi H
            projected_pts = cv2.perspectiveTransform(src_inliers, H)
            
            # Hitung Euclid Distance Error
            errors = np.linalg.norm(dst_inliers - projected_pts, axis=2)
            rmse = float(np.sqrt(np.mean(errors ** 2)))
        else:
            rmse = 999.0
    else:
        inliers_cnt = 0
        rmse = 999.0

    # 4. Precision Ratio (%)
    precision_ratio = (inliers_cnt / total_matches * 100.0) if total_matches > 0 else 0.0

    # 5. Evaluasi Kualitas Registrasi Citra (SSIM setelah Warping Geometris)
    ssim_val = 0.0
    if H is not None and inliers_cnt >= 4:
        h, w = gray1.shape
        # Lakukan warping pada citra kedua agar sejajar dengan citra acuan (img1)
        warped_gray2 = cv2.warpPerspective(gray2, H, (w, h))

        # Buat mask area tumpang tindih (overlapping region)
        overlap_mask = cv2.warpPerspective(np.ones_like(gray2, dtype=np.uint8) * 255, H, (w, h)) > 0

        if np.sum(overlap_mask) > 100:  # Memastikan area overlap cukup besar
            score, _ = ssim(gray1, warped_gray2, full=True)
            ssim_val = float(score)

    return kp1_cnt, kp2_cnt, total_matches, inliers_cnt, round(precision_ratio, 2), round(rmse, 4), round(ssim_val, 4)


def run_registration_quality_evaluation(base_path, scenarios):
    all_details = []

    print("=" * 135)
    print(f"{'Skenario':<10} | {'Metode':<10} | {'Pasangan Frame':<22} | {'Matches':<8} | {'Inliers':<8} | {'Precision (%)':<12} | {'RMSE (px)':<10} | {'SSIM':<8}")
    print("=" * 135)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', 
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')

    for scn in scenarios:
        methods_folders = {
            'Original': os.path.join(base_path, scn, "input"),
            'GHE': os.path.join(base_path, scn, "output_ghe"),
            'CLAHE': os.path.join(base_path, scn, "output_clahe"),
            'LogCh': os.path.join(base_path, scn, "output_logch")
        }

        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(methods_folders['Original'], ext)))
        image_files = sorted(list(set(image_files)))

        if len(image_files) < 2:
            print(f"[{scn.upper()}] Peringatan: Butuh minimal 2 berkas citra di folder input.")
            continue

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
                    
                    kp1_cnt, kp2_cnt, total_m, inliers, precision, rmse, ssim_score = evaluate_registration_metrics(img1, img2)
                    avg_kp = round((kp1_cnt + kp2_cnt) / 2.0, 1)
                    
                    print(f"{scn:<10} | {method_name:<10} | {pair_label:<22} | {total_m:<8} | {inliers:<8} | {precision:<12.2f} | {rmse:<10.4f} | {ssim_score:<8.4f}")

                    all_details.append({
                        'Skenario': scn,
                        'Metode': method_name,
                        'Pasangan_Frame': pair_label,
                        'Avg_Keypoints': avg_kp,
                        'Total_Matches': total_m,
                        'Valid_Matches_Inliers': inliers,
                        'Precision_Ratio (%)': precision,
                        'RMSE (Pixel)': rmse,
                        'SSIM_Registration': ssim_score
                    })

    print("=" * 135)
    return all_details

# --- Eksekusi Program Utama ---

base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"]

# Jalankan Evaluasi Metrik Kualitas Registrasi dan Pencocokan Fitur
registration_data = run_registration_quality_evaluation(base_path, scenarios)

# --- Olah Data dan Ekspor Hasil ke Berkas CSV ---
if registration_data:
    df_detail = pd.DataFrame(registration_data)
    
    # 1. Simpan Rekap Data Detail
    csv_detail_path = os.path.join(base_path, "metrik_kualitas_registrasi_detail.csv")
    df_detail.to_csv(csv_detail_path, index=False)

    # 2. Hitung Ringkasan Rata-Rata per Skenario dan Metode
    df_summary = df_detail.groupby(['Skenario', 'Metode'])[[
        'Avg_Keypoints', 
        'Total_Matches', 
        'Valid_Matches_Inliers', 
        'Precision_Ratio (%)',
        'RMSE (Pixel)',
        'SSIM_Registration'
    ]].mean().round(4).reset_index()

    csv_summary_path = os.path.join(base_path, "metrik_kualitas_registrasi_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)

    print("\n" + "=" * 95)
    print("   RINGKASAN METRIK KUALITAS REGISTRASI DAN PENCOCOKAN FITUR (RATA-RATA)   ")
    print("=" * 95)
    print(df_summary.to_string(index=False))
    print("=" * 95)
    print(f"\n[SUKSES] Rekap data detail disimpan ke   : {csv_detail_path}")
    print(f"[SUKSES] Ringkasan rata-rata disimpan ke : {csv_summary_path}")