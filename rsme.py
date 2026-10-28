import cv2
import numpy as np
import os
import glob

def calculate_temporal_rmse(folder_path):
    """
    Menghitung rata-rata RMSE antara bingkai yang berurutan dalam sebuah folder.
    """
    # Ambil semua file gambar
    img_files = sorted(glob.glob(os.path.join(folder_path, "*.[jp][pn][g]")))
    
    if len(img_files) < 2:
        return 0.0

    rmse_values = []

    for i in range(len(img_files) - 1):
        # Baca bingkai saat ini dan bingkai berikutnya
        img1 = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_files[i+1], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue

        # Pastikan ukuran sama (untuk menghindari error jika ada perbedaan dimensi)
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Hitung Mean Squared Error (MSE)
        # Konversi ke float untuk menghindari overflow pada uint8
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # Hitung RMSE
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

    # Kembalikan rata-rata RMSE dari seluruh sekuens
    if not rmse_values:
        return 0.0
    return np.mean(rmse_values)

# --- Eksekusi Perhitungan untuk Tabel 1 ---

base_path = "data_penelitian/"
scenarios = ["cctv", "medic", "under"]
methods = ["input", "output_ghe"] # Bandingkan data asli vs hasil GHE

print(f"{'Skenario':<15} | {'Metode':<15} | {'Rata-rata RMSE':<15}")
print("-" * 50)

for scn in scenarios:
    for meth in methods:
        folder = os.path.join(base_path, scn, meth)
        if os.path.exists(folder):
            avg_rmse = calculate_temporal_rmse(folder)
            # Normalisasi ke skala 0-1 (asumsi 8-bit image 0-255) agar sesuai narasi paper Anda
            normalized_rmse = avg_rmse / 255.0
            print(f"{scn:<15} | {meth:<15} | {normalized_rmse:.4f}")