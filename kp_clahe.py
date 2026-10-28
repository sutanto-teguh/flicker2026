import cv2
import numpy as np
import os
import glob
import pandas as pd

def count_sift_keypoints(image_path):
    """
    Membaca citra dari jalur file dan menghitung total keypoint SIFT.
    """
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

def process_sift_analysis_clahe(base_path, scenarios):
    all_results = []

    print("=" * 80)
    print(f"{'Skenario':<12} | {'Filename':<15} | {'KP Original':<12} | {'KP CLAHE':<12} | {'Selisih':<15}")
    print("=" * 80)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', 
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')

    for scn in scenarios:
        input_dir = os.path.join(base_path, scn, "input")
        clahe_dir = os.path.join(base_path, scn, "output_clahe")

        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files = sorted(list(set(image_files)))

        if not image_files:
            print(f"[{scn.upper()}] Peringatan: Tidak ada berkas ditemukan di '{input_dir}'")
            continue

        for img_path in image_files:
            filename = os.path.basename(img_path)
            clahe_img_path = os.path.join(clahe_dir, filename)

            # 1. Hitung Keypoint pada Citra Original
            kp_orig_count = count_sift_keypoints(img_path)

            # 2. Hitung Keypoint pada Citra Hasil CLAHE
            if os.path.exists(clahe_img_path):
                kp_clahe_count = count_sift_keypoints(clahe_img_path)
            else:
                kp_clahe_count = 0
                print(f"Peringatan: Berkas {filename} tidak ditemukan di {clahe_dir}")

            # 3. Hitung Selisih dan Persentase Perubahan
            diff = kp_clahe_count - kp_orig_count
            pct_change = (diff / kp_orig_count * 100) if kp_orig_count > 0 else 0

            print(f"{scn:<12} | {filename:<15} | {kp_orig_count:<12} | {kp_clahe_count:<12} | {diff:+d} ({pct_change:+.2f}%)")

            all_results.append({
                'Skenario': scn,
                'Filename': filename,
                'KP_Original': kp_orig_count,
                'KP_CLAHE': kp_clahe_count,
                'Selisih': diff,
                'Persentase_Perubahan (%)': round(pct_change, 2)
            })

    print("=" * 80)
    return all_results

# --- Eksekusi Program ---

base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"]

# Jalankan analisis SIFT khusus CLAHE
data_keypoints = process_sift_analysis_clahe(base_path, scenarios)

# --- Olah Data dan Simpan ke CSV ---
if data_keypoints:
    df = pd.DataFrame(data_keypoints)
    
    # 1. Simpan Data Detail per Gambar
    csv_detail_path = os.path.join(base_path, "analisis_keypoints_clahe_detail.csv")
    df.to_csv(csv_detail_path, index=False)
    
    # 2. Hitung Rata-Rata per Skenario untuk Tabel Paper
    df_summary = df.groupby('Skenario')[['KP_Original', 'KP_CLAHE', 'Selisih']].mean().reset_index()
    csv_summary_path = os.path.join(base_path, "analisis_keypoints_clahe_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)

    print("\n--- RATA-RATA TOTAL KEYPOINT SIFT PER SKENARIO (CLAHE) ---")
    print(df_summary.to_string(index=False))
    print(f"\n[SUKSES] Data detail disimpan ke   : {csv_detail_path}")
    print(f"[SUKSES] Ringkasan rata-rata disimpan ke: {csv_summary_path}")