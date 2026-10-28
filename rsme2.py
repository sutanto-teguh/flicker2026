import cv2
import numpy as np
import os
import glob

def calculate_temporal_rmse_debug(folder_path):
    # 1. Cari berbagai ekstensi gambar (huruf kecil & besar)
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Hapus duplikat jalur file jika ada, lalu urutkan
    img_files = sorted(list(set(img_files)))

    print(f"\n--- Memeriksa Folder: {folder_path} ---")
    print(f"Jumlah file ditemukan: {len(img_files)}")

    # 2. Cek apakah gambar cukup untuk dihitung pasangan temporalnya
    if len(img_files) < 2:
        print("❌ PENYEBAB RSME 0: Gambar kurang dari 2 atau folder tidak ditemukan!")
        return 0.0

    rmse_values = []

    for i in range(len(img_files) - 1):
        # Baca gambar secara Grayscale
        img1 = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_files[i+1], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print(f"❌ Gagal membaca gambar: {img_files[i]} atau {img_files[i+1]}")
            continue

        # Samakan ukuran jika beda dimensi
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Hitung MSE (Mean Squared Error)
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # Hitung RMSE
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

    if not rmse_values:
        print("❌ Tidak ada pasangan gambar yang berhasil dihitung!")
        return 0.0

    avg_rmse = np.mean(rmse_values)
    
    # Cek jika ternyata gambarnya memang identik
    if avg_rmse == 0.0:
        print("⚠️ PERINGATAN: Nilai RMSE 0 karena semua gambar dalam folder ini SAMA PERSIS.")

    return avg_rmse

# --- EKSEKUSI PEMERIKSAAN ---

base_path = "data_penelitian/"
scenarios = ["cctv", "medic", "under"]
methods = ["input", "output_ghe"]

print(f"{'Skenario':<12} | {'Metode':<12} | {'RMSE (0-1)':<12}")
print("=" * 45)

for scn in scenarios:
    for meth in methods:
        folder = os.path.join(base_path, scn, meth)
        if os.path.exists(folder):
            avg_rmse = calculate_temporal_rmse_debug(folder)
            # Normalisasi ke skala 0 - 1 (dibagi 255)
            normalized_rmse = avg_rmse / 255.0
            print(f"RESULTS -> {scn:<10} | {meth:<10} | {normalized_rmse:.4f}\n")
        else:
            print(f"❌ Folder tidak ditemukan: {folder}\n")