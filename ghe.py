import cv2
import numpy as np
import os
import glob

def apply_global_he(image):
    """
    Menerapkan Global Histogram Equalization pada citra berwarna.
    Dilakukan pada kanal Y (Luminance) dalam ruang warna YCrCb 
    agar tidak merusak keseimbangan warna asli.
    """
    # Konversi dari BGR ke YCrCb
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Pisahkan kanal
    y, cr, cb = cv2.split(ycrcb_img)
    
    # Terapkan Global Histogram Equalization pada kanal Y
    y_eq = cv2.equalizeHist(y)
    
    # Gabungkan kembali kanal
    updated_ycrcb = cv2.merge((y_eq, cr, cb))
    
    # Konversi kembali ke BGR
    result = cv2.cvtColor(updated_ycrcb, cv2.COLOR_YCrCb2BGR)
    return result

def process_flicker_reduction(scenario_name, input_folder, output_folder):
    print(f"--- Memproses Skenario: {scenario_name} ---")
    
    # Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ambil semua file gambar (jpg/png)
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.[jp][pn][g]")))
    
    if not image_files:
        print(f"Peringatan: Tidak ada gambar ditemukan di {input_folder}")
        return

    for img_path in image_files:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # Proses Global HE
        processed_img = apply_global_he(img)
        
        # Simpan hasil
        cv2.imwrite(os.path.join(output_folder, filename), processed_img)
        
    print(f"Selesai! Hasil disimpan di: {output_folder}\n")

# --- Eksekusi Program ---

# Definisikan path folder (Sesuaikan dengan direktori Anda)
base_path = "data_penelitian/"
scenarios = ["cctv", "medical", "underwater"]

for scn in scenarios:
    input_dir = os.path.join(base_path, scn, "input")
    output_dir = os.path.join(base_path, scn, "output_ghe")
    
    process_flicker_reduction(scn, input_dir, output_dir)