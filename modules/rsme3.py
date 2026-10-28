import cv2
import numpy as np
import os
import glob

def calculate_temporal_rmse(folder_path):
    """
    Menghitung rata-rata RMSE temporal antar-bingkai berurutan.
    Mendukung berbagai variasi ekstensi huruf besar/kecil.
    """
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP')
    img_files = []
    
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Eliminasi duplikat jalur file dan urutkan
    img_files = sorted(list(set(img_files)))

    # Validasi kecukupan bingkai
    if len(img_files) < 2:
        return None

    rmse_values = []

    for i in range(len(img_files) - 1):
        img1 = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_files[i+1], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue

        # Penyelarasan dimensi citra
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Perhitungan MSE dan RMSE
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

    if not rmse_values:
        return None

    return np.mean(rmse_values)

# --- EKSEKUSI PEMROSESAN SELURUH DATASET ---

base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"]
methods = ["input", "output_logch"]  # Tambahkan "output_logch" jika metode Anda sudah selesai di-running

print(f"{'Skenario':<12} | {'Metode':<12} | {'RMSE (Skala 0-255)':<20} | {'RMSE Normalisasi (0-1)':<20}")
print("=" * 70)

for scn in scenarios:
    for mth in methods:
        folder = os.path.join(base_path, scn, mth)
        
        if os.path.exists(folder):
            avg_rmse = calculate_temporal_rmse(folder)
            
            if avg_rmse is not None:
                norm_rmse = avg_rmse / 255.0
                print(f"{scn:<12} | {mth:<12} | {avg_rmse:<20.4f} | {norm_rmse:<20.4f}")
            else:
                print(f"{scn:<12} | {mth:<12} | {'Gambar < 2':<20} | {'N/A':<20}")
        else:
            print(f"{scn:<12} | {mth:<12} | {'Folder Tidak Ada':<20} | {'N/A':<20}")
