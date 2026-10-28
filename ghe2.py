import cv2
import numpy as np
import os
import glob

def apply_global_he(image):
    """
    Menerapkan Global Histogram Equalization (GHE).
    Mendukung citra Grayscale (1 channel) dan Berwarna/BGR (3 channel).
    """
    # Deteksi citra Grayscale (umum pada data medis seperti x-ray/fluoroskopi)
    if len(image.shape) == 2 or image.shape[2] == 1:
        return cv2.equalizeHist(image)
    
    # Deteksi citra Berwarna / BGR (seperti CCTV dan Underwater)
    # Konversi ke ruang warna YCrCb agar pemrosesan intensitas tidak merusak warna
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    
    # Ekualisasi hanya dilakukan pada kanal Luminance (Y)
    y_eq = cv2.equalizeHist(y)
    
    # Gabungkan kembali dan konversi ke BGR
    updated_ycrcb = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(updated_ycrcb, cv2.COLOR_YCrCb2BGR)

def process_flicker_reduction(scenario_name, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Dukungan format file luas (termasuk .tif/.tiff yang sering digunakan pada medis)
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', 
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"[{scenario_name.upper()}] Peringatan: Tidak ada gambar ditemukan di '{input_folder}'")
        return

    print(f"[{scenario_name.upper()}] Memproses {len(image_files)} gambar...")
    processed_count = 0

    for img_path in image_files:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            continue
            
        # Konversi jika citra memiliki 4 kanal (RGBA) menjadi 3 kanal (BGR)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        processed_img = apply_global_he(img)
        cv2.imwrite(os.path.join(output_folder, filename), processed_img)
        processed_count += 1
        
    print(f"[{scenario_name.upper()}] Selesai! {processed_count} gambar disimpan di: '{output_folder}'\n")

# --- Eksekusi Program ---
base_path = "data_penelitian"
scenarios = ["cctv", "medical", "under"]

for scn in scenarios:
    input_dir = os.path.join(base_path, scn, "input")
    output_dir = os.path.join(base_path, scn, "output_ghe")
    process_flicker_reduction(scn, input_dir, output_dir)