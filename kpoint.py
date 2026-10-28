import cv2
import numpy as np
import os
import glob
import pandas as pd

def count_sift_keypoints(image_path):
    """
    Membaca citra dan menghitung total keypoint SIFT.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0, None
    
    # Konversi ke Grayscale untuk ekstraksi SIFT
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Inisialisasi SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return len(keypoints), keypoints

def process_sift_analysis(base_path, scenarios):
    all_results = []

    print("=" * 75)
    print(f"{'Skenario':<12} | {'Filename':<15} | {'KP Original':<12} | {'KP GHE':<12} | {'Selisih':<10}")
    print("=" * 75)

    for scn in scenarios:
        input_dir = os.path.join(base_path, scn, "input")
        ghe_dir = os.path.join(base_path, scn, "output_ghe")

        image_files = sorted(glob.glob(os.path.join(input_dir, "*.[jp][pn][g]")))

        if not image_files:
            print(f"Peringatan: Tidak ada berkas ditemukan di {input_dir}")
            continue

        for img_path in image_files:
            filename = os.path.basename(img_path)
            ghe_img_path = os.path.join(ghe_dir, filename)

            # 1. Hitung Keypoint pada Citra Original
            kp_orig_count, _ = count_sift_keypoints(img_path)

            # 2. Hitung Keypoint pada Citra Hasil GHE
            if os.path.exists(ghe_img_path):
                kp_ghe_count, _ = count_sift_keypoints(ghe_img_path)
            else:
                kp_ghe_count = 0
                print(f"Berkas {filename} tidak ditemukan di {ghe_dir}")

            # 3. Hitung Selisih dan Persentase
            diff = kp_ghe_count - kp_orig_count
            pct_change = (diff / kp_orig_count * 100) if kp_orig_count > 0 else 0

            print(f"{scn:<12} | {filename:<15} | {kp_orig_count:<12} | {kp_ghe_count:<12} | {diff:+d} ({pct_change:+.1f}%)")

            all_results.append({
                'Skenario': scn,
                'Filename': filename,
                'KP_Original': kp_orig_count,
                'KP_GHE': kp_ghe_count,
                'Selisih': diff,
                'Persentase_Perubahan (%)': round(pct_change, 2)
            })

    print("=" * 75)
    return all_results

# --- Eksekusi Program ---

base_path = "data_penelitian/"
scenarios = ["cctv", "medical", "underwater"]

# Jalankan analisis SIFT
data_keypoints = process_sift_analysis(base_path, scenarios)

# --- Olah Data dan Simpan ke CSV ---
if data_keypoints:
    df = pd.DataFrame(data_keypoints)
    
    # 1. Simpan Data Detail
    csv_detail_path = os.path.join(base_path, "analisis_keypoints_sift_detail.csv")
    df.to_csv(csv_detail_path, index=False)
    
    # 2. Hitung Rata-Rata per Skenario untuk Tabel Paper
    df_summary = df.groupby('Skenario')[['KP_Original', 'KP_GHE', 'Selisih']].mean().reset_index()
    csv_summary_path = os.path.join(base_path, "analisis_keypoints_sift_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)

    print("\n--- RATA-RATA TOTAL KEYPOINT PER SKENARIO ---")
    print(df_summary.to_string(index=False))
    print(f"\n[SUKSES] Data detail disimpan ke: {csv_detail_path}")
    print(f"[SUKSES] Ringkasan rata-rata disimpan ke: {csv_summary_path}")