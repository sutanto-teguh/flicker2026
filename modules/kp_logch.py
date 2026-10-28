import cv2
import numpy as np
import os
import glob
import pandas as pd

def count_sift_keypoints(image_path):
    """
    Membaca citra dari jalur file dan menghitung total keypoint SIFT.
    Jika berkas tidak ditemukan, mengembalikan 0.
    """
    if not os.path.exists(image_path):
        return 0

    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    # Konversi ke Grayscale untuk ekstraksi SIFT
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    return len(keypoints)

def process_all_methods_sift_analysis(base_path, scenarios):
    all_results = []

    print("=" * 95)
    print(f"{'Skenario':<10} | {'Filename':<12} | {'Original':<10} | {'GHE':<10} | {'CLAHE':<10} | {'Log-Ch':<10}")
    print("=" * 95)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', 
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')

    for scn in scenarios:
        input_dir = os.path.join(base_path, scn, "input")
        ghe_dir = os.path.join(base_path, scn, "output_ghe")
        clahe_dir = os.path.join(base_path, scn, "output_clahe")
        logch_dir = os.path.join(base_path, scn, "output_logch")

        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files = sorted(list(set(image_files)))

        if not image_files:
            print(f"[{scn.upper()}] Peringatan: Tidak ada berkas ditemukan di '{input_dir}'")
            continue

        for img_path in image_files:
            filename = os.path.basename(img_path)

            ghe_img_path = os.path.join(ghe_dir, filename)
            clahe_img_path = os.path.join(clahe_dir, filename)
            logch_img_path = os.path.join(logch_dir, filename)

            # 1. Hitung Keypoint SIFT untuk setiap folder metode
            kp_orig = count_sift_keypoints(img_path)
            kp_ghe = count_sift_keypoints(ghe_img_path)
            kp_clahe = count_sift_keypoints(clahe_img_path)
            kp_logch = count_sift_keypoints(logch_img_path)

            print(f"{scn:<10} | {filename:<12} | {kp_orig:<10} | {kp_ghe:<10} | {kp_clahe:<10} | {kp_logch:<10}")

            all_results.append({
                'Skenario': scn,
                'Filename': filename,
                'KP_Original': kp_orig,
                'KP_GHE': kp_ghe,
                'KP_CLAHE': kp_clahe,
                'KP_LogCh': kp_logch
            })

    print("=" * 95)
    return all_results

# --- Eksekusi Program ---

base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"]

# Jalankan analisis SIFT untuk semua metode
data_keypoints = process_all_methods_sift_analysis(base_path, scenarios)

# --- Olah Data dan Simpan ke CSV ---
if data_keypoints:
    df = pd.DataFrame(data_keypoints)
    
    # 1. Simpan Data Detail per Gambar
    csv_detail_path = os.path.join(base_path, "komparasi_keypoints_sift_detail.csv")
    df.to_csv(csv_detail_path, index=False)
    
    # 2. Hitung Rata-Rata per Skenario untuk Tabel Paper
    df_summary = df.groupby('Skenario')[['KP_Original', 'KP_GHE', 'KP_CLAHE', 'KP_LogCh']].mean().reset_index()
    csv_summary_path = os.path.join(base_path, "komparasi_keypoints_sift_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)

    print("\n--- RATA-RATA TOTAL KEYPOINT SIFT PER SKENARIO (SEMUA METODE) ---")
    print(df_summary.to_string(index=False))
    print(f"\n[SUKSES] Rekap data detail disimpan ke   : {csv_detail_path}")
    print(f"[SUKSES] Ringkasan rata-rata disimpan ke : {csv_summary_path}")
