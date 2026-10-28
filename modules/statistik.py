import cv2
import numpy as np
import os
import glob
import pandas as pd

def compute_feature_stats(folder_path, detector_type='sift'):
    """
    Menghitung statistik rata-rata keypoints, valid matches (inliers),
    dan precision ratio untuk sekuens gambar dalam suatu folder.
    """
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff',
            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    image_paths = sorted(list(set(image_paths)))

    if len(image_paths) < 2:
        return None

    if detector_type.lower() == 'sift':
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=3000)
        norm_type = cv2.NORM_HAMMING

    tot_kp_list, valid_matches_list, precision_list = [], [], []

    for i in range(len(image_paths) - 1):
        img1 = cv2.imread(image_paths[i])
        img2 = cv2.imread(image_paths[i+1])

        if img1 is None or img2 is None:
            continue

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            continue

        bf = cv2.BFMatcher(norm_type)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        valid_matches = 0
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                valid_matches = int(np.sum(mask.ravel()))

        total_kp = len(kp1) + len(kp2)
        precision = (valid_matches / len(good_matches) * 100.0) if len(good_matches) > 0 else 0.0

        tot_kp_list.append(total_kp)
        valid_matches_list.append(valid_matches)
        precision_list.append(precision)

    if not tot_kp_list:
        return None

    return {
        'Avg Keypoints': round(float(np.mean(tot_kp_list)), 1),
        'Avg Valid Matches': round(float(np.mean(valid_matches_list)), 1),
        'Precision (%)': round(float(np.mean(precision_list)), 2)
    }

def generate_statistics_table(base_path, detector_type='sift'):
    scenarios = ["cctv", "medic", "underwater"]
    methods = {
        "Raw (Input)": "input",
        "Global HE": "output_ghe",
        "CLAHE": "output_clahe",
        "Proposed Log-Ch": "output_logch"
    }

    table_data = []

    for scn in scenarios:
        for method_label, folder_name in methods.items():
            folder_path = os.path.join(base_path, scn, folder_name)
            
            if os.path.exists(folder_path):
                stats = compute_feature_stats(folder_path, detector_type=detector_type)
                if stats:
                    table_data.append({
                        "Skenario": scn.upper(),
                        "Metode": method_label,
                        "Avg Keypoints": stats['Avg Keypoints'],
                        "Valid Matches (Inliers)": stats['Avg Valid Matches'],
                        "Precision Ratio (%)": stats['Precision (%)']
                    })
                else:
                    table_data.append({
                        "Skenario": scn.upper(),
                        "Metode": method_label,
                        "Avg Keypoints": "N/A",
                        "Valid Matches (Inliers)": "N/A",
                        "Precision Ratio (%)": "N/A"
                    })

    df = pd.DataFrame(table_data)
    
    # Cetak ke Terminal dalam Format Markdown Table
    print("\n" + "="*85)
    print(f"TABEL STATISTIK EVALUASI FITUR ({detector_type.upper()})")
    print("="*85)
    print(df.to_markdown(index=False))
    print("="*85 + "\n")

    # Simpan ke CSV dan Excel untuk disalin ke LaTeX / MS Word
    df.to_csv("tabel_statistik_fitur.csv", index=False)
    try:
        df.to_excel("tabel_statistik_fitur.xlsx", index=False)
        print("File berhasil disimpan ke 'tabel_statistik_fitur.csv' dan 'tabel_statistik_fitur.xlsx'")
    except Exception:
        print("File berhasil disimpan ke 'tabel_statistik_fitur.csv'")

# --- EKSEKUSI PROGRAM ---
base_directory = "data_penelitian"
generate_statistics_table(base_directory, detector_type='sift')
